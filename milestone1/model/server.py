"""
server.py — Batch Inference Server with Result Caching
======================================================

Provides GPU-efficient batch inference for the RV32I transformer model
with thread-safe LRU result caching, request deduplication, and automatic
batch collection.

Classes:
    ServerConfig       -- Dataclass for server configuration parameters
    ResultCache        -- Thread-safe LRU cache with TTL eviction
    DeduplicationTracker -- Deduplicates identical in-flight requests
    _PendingRequest    -- Dataclass for a single queued inference request
    BatchCollector     -- Collects requests into batches for GPU processing
    ModelServer        -- Main server integrating cache, dedup, batching
"""

import dataclasses
import hashlib
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure milestone1 is importable
# ---------------------------------------------------------------------------
_milestone1 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _milestone1 not in sys.path:
    sys.path.insert(0, _milestone1)

from rv32i_mask import decode as rv_decode
from rv32i_mask import encode as rv_encode
from rv32i_model import RV32IFieldTokenizer, RV32ITransformer, RV32IVocab

logger = logging.getLogger(__name__)


# =====================================================================
#  SERVER CONFIG
# =====================================================================

@dataclasses.dataclass
class ServerConfig:
    """Configuration for ModelServer.

    Attributes:
        checkpoint_path:      Path to the .pt model checkpoint file.
        model_size:           Model size identifier ('small', 'medium', 'large').
        device:               Device string ('cuda', 'cpu', or 'auto').
        max_batch_size:       Maximum number of requests per GPU batch.
        batch_timeout:        Maximum seconds to wait before dispatching a
                              partial batch (float).
        cache_max_size:       Maximum number of entries in the result cache.
        cache_ttl:            Default time-to-live for cache entries (seconds).
        gpu_memory_fraction:  Fraction of GPU memory to reserve (0.0-1.0).
    """

    checkpoint_path: str = ""
    model_size: str = "large"
    device: str = "auto"
    max_batch_size: int = 16
    batch_timeout: float = 0.05
    cache_max_size: int = 1024
    cache_ttl: float = 300.0
    gpu_memory_fraction: float = 0.9


# =====================================================================
#  RESULT CACHE
# =====================================================================

class ResultCache:
    """Thread-safe LRU cache with per-entry TTL eviction.

    Stores generated token sequences keyed by a deterministic hash of the
    prompt tokens and generation parameters.  Entries that exceed their TTL
    are lazily evicted on access.
    """

    def __init__(self, max_size: int = 1024, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl

        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}          # key -> value
        self._expiry: dict[str, float] = {}        # key -> timestamp
        self._access_order: list[str] = []          # LRU order (most recent at end)

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # -----------------------------------------------------------------
    #  Key generation
    # -----------------------------------------------------------------

    @staticmethod
    def make_key(
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None = None,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Build a deterministic SHA-256 cache key from generation params.

        Args:
            prompt_tokens:        Tokenized prompt (list of ints).
            max_new_tokens:       Max tokens to generate.
            temperature:          Sampling temperature.
            top_k:                Top-k sampling parameter (None = disabled).
            top_p:                Top-p nucleus sampling parameter.
            repetition_penalty:   Repetition penalty factor.

        Returns:
            64-character hex digest string.
        """
        parts = (
            str(prompt_tokens),
            str(max_new_tokens),
            f"{temperature:.6f}",
            str(top_k),
            str(top_p) if top_p is not None else "None",
            f"{repetition_penalty:.6f}",
        )
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` on miss / expiry."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Check TTL
            expiry = self._expiry.get(key)
            if expiry is not None and time.monotonic() > expiry:
                self._evict_locked(key)
                self._misses += 1
                return None

            # Move to end of LRU list (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store *value* under *key* with an optional TTL override.

        If the cache is at capacity, the least-recently-used entry is evicted.
        """
        if ttl is None:
            ttl = self.default_ttl

        with self._lock:
            # If key already exists, remove it first to avoid duplicates
            if key in self._cache:
                self._access_order.remove(key)

            # Evict LRU entries if at capacity
            while len(self._cache) >= self.max_size and self._access_order:
                lru_key = self._access_order[0]
                self._evict_locked(lru_key)

            self._cache[key] = value
            self._expiry[key] = time.monotonic() + ttl
            self._access_order.append(key)

    def invalidate(self, key: str) -> bool:
        """Remove a specific cache entry.  Returns True if it existed."""
        with self._lock:
            if key not in self._cache:
                return False
            self._evict_locked(key)
            return True

    def clear(self) -> None:
        """Remove all cache entries and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> dict[str, Any]:
        """Return a dict with cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "evictions": self._evictions,
            }

    # -----------------------------------------------------------------
    #  Internal helpers
    # -----------------------------------------------------------------

    def _evict_locked(self, key: str) -> None:
        """Evict a single entry.  Must be called with *self._lock* held."""
        self._cache.pop(key, None)
        self._expiry.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)
        self._evictions += 1


# =====================================================================
#  DEDUPLICATION TRACKER
# =====================================================================

class DeduplicationTracker:
    """Tracks in-flight requests so identical prompts share one GPU run.

    When a request arrives whose deduplication key matches one already in
    flight, the caller receives the list of existing waiters.  Once the
    original request completes, all waiters are notified and receive the
    same result.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._inflight: dict[str, list[_PendingRequest]] = {}

    def register(
        self, dedup_key: str, request: "_PendingRequest"
    ) -> list["_PendingRequest"] | None:
        """Register a new request.

        Args:
            dedup_key: Unique key identifying the logical request.
            request:   The ``_PendingRequest`` being submitted.

        Returns:
            If this is the *first* request for *dedup_key*, returns ``None``
            (the caller owns the request).  If duplicates exist, returns the
            list of existing ``_PendingRequest`` objects that will receive
            the same result.
        """
        with self._lock:
            waiters = self._inflight.get(dedup_key)
            if waiters is not None:
                waiters.append(request)
                return waiters
            self._inflight[dedup_key] = []
            return None

    def complete(self, dedup_key: str) -> list["_PendingRequest"]:
        """Mark a deduplication key as completed and return all waiters.

        The caller is responsible for signaling all returned waiters with the
        shared result.
        """
        with self._lock:
            waiters = self._inflight.pop(dedup_key, [])
            return waiters


# =====================================================================
#  PENDING REQUEST
# =====================================================================

@dataclasses.dataclass
class _PendingRequest:
    """A single inference request waiting to be batched and executed."""

    prompt_tokens: list[int]
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    repetition_penalty: float
    event: threading.Event = dataclasses.field(default_factory=threading.Event)
    result: list[int] | None = None
    error: Exception | None = None


# =====================================================================
#  BATCH COLLECTOR
# =====================================================================

class BatchCollector:
    """Collects individual inference requests into GPU-efficient batches.

    Requests are submitted with ``submit()``, which returns a
    ``_PendingRequest`` whose ``event`` blocks until the result is ready.
    A background thread periodically drains the pending queue and runs the
    model on the collected batch.
    """

    def __init__(
        self,
        model: RV32ITransformer,
        tokenizer: RV32IFieldTokenizer,
        device: torch.device,
        max_batch_size: int = 16,
        batch_timeout: float = 0.05,
        max_seq_len: int = 1024,
        cache: ResultCache | None = None,
        dedup_tracker: DeduplicationTracker | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.max_seq_len = max_seq_len
        self.cache = cache
        self.dedup_tracker = dedup_tracker

        self._lock = threading.Lock()
        self._pending: list[_PendingRequest] = []
        self._running = False
        self._thread: threading.Thread | None = None

    # -----------------------------------------------------------------
    #  Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> None:
        """Start the background batch-collection thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._collect_loop, daemon=True, name="BatchCollector"
        )
        self._thread.start()
        print("[Server] BatchCollector background thread started")

    def stop(self) -> None:
        """Stop the background thread and process any remaining requests."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Drain any remaining requests as single-item batches
        self._flush_pending()
        print("[Server] BatchCollector stopped")

    def pending_count(self) -> int:
        """Return the number of requests waiting to be batched."""
        with self._lock:
            return len(self._pending)

    # -----------------------------------------------------------------
    #  Submission
    # -----------------------------------------------------------------

    def submit(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        repetition_penalty: float,
    ) -> _PendingRequest:
        """Submit a single inference request for batched execution.

        Args:
            prompt_tokens:        Tokenized prompt.
            max_new_tokens:       Maximum tokens to generate.
            temperature:          Sampling temperature.
            top_k:                Top-k sampling (None = disabled).
            top_p:                Top-p nucleus sampling (None = disabled).
            repetition_penalty:   Repetition penalty factor.

        Returns:
            ``_PendingRequest`` whose ``event`` blocks until the result is
            ready.  Check ``.result`` or ``.error`` after ``event.wait()``.
        """
        request = _PendingRequest(
            prompt_tokens=list(prompt_tokens),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        with self._lock:
            self._pending.append(request)

        return request

    # -----------------------------------------------------------------
    #  Background collection loop
    # -----------------------------------------------------------------

    def _collect_loop(self) -> None:
        """Background thread: drain pending requests and run batched inference."""
        while self._running:
            time.sleep(self.batch_timeout)
            self._drain_batch()

    def _drain_batch(self) -> None:
        """Take up to max_batch_size requests and run them as a batch."""
        batch: list[_PendingRequest] = []

        with self._lock:
            n = min(self.max_batch_size, len(self._pending))
            batch = self._pending[:n]
            self._pending = self._pending[n:]

        if not batch:
            return

        self._run_batch(batch)

    def _flush_pending(self) -> None:
        """Process all remaining pending requests (called on shutdown)."""
        while True:
            batch: list[_PendingRequest] = []
            with self._lock:
                n = min(self.max_batch_size, len(self._pending))
                if n == 0:
                    break
                batch = self._pending[:n]
                self._pending = self._pending[n:]
            self._run_batch(batch)

    # -----------------------------------------------------------------
    #  Batch execution
    # -----------------------------------------------------------------

    def _run_batch(self, batch: list[_PendingRequest]) -> None:
        """Pad, run model.generate(), slice results, and signal events.

        Handles CUDA OOM by splitting the batch in half and retrying.
        Falls back to single inference on total batch failure.
        """
        if not batch:
            return

        try:
            self._execute_batch(batch)
        except torch.cuda.OutOfMemoryError:
            print(
                f"[Server] CUDA OOM with batch size {len(batch)}, "
                f"splitting and retrying"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(batch) > 1:
                mid = len(batch) // 2
                self._run_batch(batch[:mid])
                self._run_batch(batch[mid:])
            else:
                self._signal_error(batch[0], RuntimeError("CUDA OOM on single request"))
        except Exception as e:
            print(f"[Server] Batch execution error: {e}")
            if len(batch) > 1:
                # Fallback: run each request individually
                for req in batch:
                    self._run_single_fallback(req)
            else:
                self._signal_error(batch[0], e)

    def _execute_batch(self, batch: list[_PendingRequest]) -> None:
        """Core batched generation logic.

        Pads all prompts to the same length, runs a single forward pass,
        then slices the generated tokens for each request.
        """
        batch_size = len(batch)

        # Determine max prompt length in the batch
        max_prompt_len = max(len(r.prompt_tokens) for r in batch)

        # Pad all prompts to the same length
        padded: list[list[int]] = []
        for req in batch:
            pad_len = max_prompt_len - len(req.prompt_tokens)
            padded.append(
                [RV32IVocab.PAD] * pad_len + req.prompt_tokens
            )

        input_ids = torch.tensor(
            padded, dtype=torch.long, device=self.device
        )

        # All requests in a batch must share the same generation parameters
        # for a single model.generate() call.  We use the parameters from
        # the first request.
        ref = batch[0]
        max_new = ref.max_new_tokens
        temperature = ref.temperature
        top_k = ref.top_k

        # Run generation under no_grad
        with torch.no_grad():
            self.model.eval()
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new,
                temperature=temperature if temperature > 0 else 1.0,
                top_k=top_k if top_k is not None else min(40, RV32IVocab.VOCAB_SIZE),
                eos_token_id=RV32IVocab.SEP,
            )

        # Clean up CUDA cache after batch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Slice results: output[:, max_prompt_len:] gives the generated tokens
        # for each batch item.
        for i, req in enumerate(batch):
            full_seq = output[i].tolist()

            # Remove leading PAD tokens
            gen_start = max_prompt_len - len(req.prompt_tokens)
            generated = full_seq[gen_start:]

            req.result = generated

            # Store in cache if available
            if self.cache is not None:
                cache_key = ResultCache.make_key(
                    req.prompt_tokens,
                    req.max_new_tokens,
                    req.temperature,
                    req.top_k,
                    req.top_p,
                    req.repetition_penalty,
                )
                self.cache.set(cache_key, generated)

            req.event.set()

    def _run_single_fallback(self, req: _PendingRequest) -> None:
        """Run a single request outside of batching as a fallback."""
        try:
            input_ids = torch.tensor(
                [req.prompt_tokens], dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                self.model.eval()
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature if req.temperature > 0 else 1.0,
                    top_k=req.top_k if req.top_k is not None else min(40, RV32IVocab.VOCAB_SIZE),
                    eos_token_id=RV32IVocab.SEP,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            generated = output[0].tolist()
            req.result = generated

            if self.cache is not None:
                cache_key = ResultCache.make_key(
                    req.prompt_tokens,
                    req.max_new_tokens,
                    req.temperature,
                    req.top_k,
                    req.top_p,
                    req.repetition_penalty,
                )
                self.cache.set(cache_key, generated)

            req.event.set()
        except Exception as e:
            self._signal_error(req, e)

    @staticmethod
    def _signal_error(req: _PendingRequest, error: Exception) -> None:
        """Set the error on a request and signal its event."""
        req.error = error
        req.event.set()


# =====================================================================
#  MODEL SERVER
# =====================================================================

class ModelServer:
    """Main server integrating cache, deduplication, and batched inference.

    Usage::

        config = ServerConfig(checkpoint_path="checkpoints/checkpoint_best.pt")
        server = ModelServer(config)
        server.start()

        result = server.infer(
            prompt_tokens=[CLS, ADD_TOKEN, ...],
            max_new_tokens=81,
        )
        print(result)

        server.stop()
    """

    def __init__(self, config: ServerConfig):
        self.config = config

        # Resolve device
        if config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(config.device)

        # Load model
        self.model: RV32ITransformer = self._load_model(config.checkpoint_path)
        self.tokenizer = RV32IFieldTokenizer()

        # Initialize components
        self.cache = ResultCache(
            max_size=config.cache_max_size,
            default_ttl=config.cache_ttl,
        )
        self.dedup_tracker = DeduplicationTracker()
        self.batch_collector = BatchCollector(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_batch_size=config.max_batch_size,
            batch_timeout=config.batch_timeout,
            max_seq_len=self.model.max_seq_len,
            cache=self.cache,
            dedup_tracker=self.dedup_tracker,
        )

        self._running = False
        print(
            f"[Server] ModelServer initialized "
            f"(device={self.device}, cache_size={config.cache_max_size}, "
            f"batch_size={config.max_batch_size})"
        )

    # -----------------------------------------------------------------
    #  Model loading
    # -----------------------------------------------------------------

    def _load_model(self, checkpoint_path: str) -> RV32ITransformer:
        """Load model from checkpoint with architecture migration."""
        # Reuse PrefixGenerator's loading logic inline to avoid a circular
        # dependency.  This imports the helper functions.
        _dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if _dir not in sys.path:
            sys.path.insert(0, _dir)

        try:
            from prefix_generator import (
                _build_key_map,
                _infer_model_dims,
            )

            if not os.path.isfile(checkpoint_path):
                logger.warning(
                    f"Checkpoint not found: {checkpoint_path}, "
                    f"creating fresh model"
                )
                return self._create_fresh_model()

            ckpt = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state_dict = ckpt.get("model_state_dict", ckpt)

            dims = _infer_model_dims(state_dict)
            logger.info(
                f"Inferred model dims: d_model={dims['d_model']}, "
                f"n_layers={dims['n_layers']}, d_ff={dims['d_ff']}, "
                f"vocab_size={dims['vocab_size']}"
            )

            model = RV32ITransformer(
                vocab_size=dims["vocab_size"],
                d_model=dims["d_model"],
                n_layers=dims["n_layers"],
                n_heads=dims["n_heads"],
                d_ff=dims["d_ff"],
                max_seq_len=dims["max_seq_len"],
                dropout=0.0,
                tie_weights=True,
                use_rope=True,
            )

            key_map = _build_key_map(dims["n_layers"])
            mapped_sd = {}
            unmapped_old = []
            for old_key, tensor in state_dict.items():
                if old_key in key_map:
                    mapped_sd[key_map[old_key]] = tensor
                else:
                    unmapped_old.append(old_key)

            if unmapped_old:
                logger.info(
                    f"Dropped {len(unmapped_old)} old keys (expected): "
                    f"{unmapped_old[:5]}{'...' if len(unmapped_old) > 5 else ''}"
                )

            result = model.load_state_dict(mapped_sd, strict=False)
            if result.missing_keys:
                logger.info(
                    f"Missing keys (new architecture, randomly initialized): "
                    f"{result.missing_keys[:5]}"
                    f"{'...' if len(result.missing_keys) > 5 else ''}"
                )

            model = model.to(self.device)
            model.eval()
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded successfully ({total_params:,} params)")
            return model

        except ImportError:
            logger.warning(
                "prefix_generator helpers not importable, "
                "creating fresh model"
            )
            return self._create_fresh_model()

    def _create_fresh_model(self) -> RV32ITransformer:
        """Create a fresh model as a fallback."""
        logger.info("Creating fresh fallback model (d_model=256, 8 layers)")
        model = RV32ITransformer(
            vocab_size=RV32IVocab.VOCAB_SIZE,
            d_model=256,
            n_layers=8,
            n_heads=8,
            d_ff=1024,
            max_seq_len=1024,
            dropout=0.0,
            tie_weights=True,
            use_rope=True,
        )
        model = model.to(self.device)
        model.eval()
        return model

    # -----------------------------------------------------------------
    #  Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> None:
        """Start the batch-collection background thread."""
        if self._running:
            return
        self.batch_collector.start()
        self._running = True
        print("[Server] ModelServer started")

    def stop(self) -> None:
        """Stop the batch-collection thread and release resources."""
        if not self._running:
            return
        self.batch_collector.stop()
        self._running = False
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("[Server] ModelServer stopped")

    # -----------------------------------------------------------------
    #  Inference
    # -----------------------------------------------------------------

    def infer(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int = 81,
        temperature: float = 0.9,
        top_k: int | None = 40,
        top_p: float | None = 0.9,
        repetition_penalty: float = 1.1,
    ) -> list[int]:
        """Run inference on a single prompt, with caching and batching.

        1. Check the result cache for a matching entry.
        2. If not cached, submit to the batch collector and wait.
        3. Return the generated token sequence.

        Args:
            prompt_tokens:        Tokenized prompt (list of int).
            max_new_tokens:       Maximum tokens to generate.
            temperature:          Sampling temperature.
            top_k:                Top-k sampling (None = disabled).
            top_p:                Top-p nucleus sampling (None = disabled).
            repetition_penalty:   Repetition penalty factor.

        Returns:
            Generated token sequence (list of int, including prompt).
        """
        if not self._running:
            raise RuntimeError(
                "ModelServer is not running.  Call start() first."
            )

        # 1. Check cache
        cache_key = ResultCache.make_key(
            prompt_tokens, max_new_tokens, temperature, top_k, top_p,
            repetition_penalty,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # 2. Check deduplication
        dedup_key = cache_key  # same as cache key for dedup
        req = _PendingRequest(
            prompt_tokens=list(prompt_tokens),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        existing_waiters = self.dedup_tracker.register(dedup_key, req)

        if existing_waiters is not None:
            # Duplicate request -- wait for the original to complete
            req.event.wait()
            if req.error is not None:
                raise req.error
            return req.result

        # 3. Submit to batch collector
        try:
            request = self.batch_collector.submit(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Wait for the batch to complete
            request.event.wait()

            if request.error is not None:
                raise request.error

            result = request.result

            # Notify dedup waiters
            waiters = self.dedup_tracker.complete(dedup_key)
            for w in waiters:
                w.result = result
                w.event.set()

            return result

        except Exception:
            # Clean up dedup entry on failure
            waiters = self.dedup_tracker.complete(dedup_key)
            for w in waiters:
                w.error = RuntimeError("Original request failed")
                w.event.set()
            raise

    def infer_batch(
        self,
        prompts: list[list[int]],
        max_new_tokens: int = 81,
        temperature: float = 0.9,
        top_k: int | None = 40,
        top_p: float | None = 0.9,
        repetition_penalty: float = 1.1,
    ) -> list[list[int]]:
        """Run inference on multiple prompts concurrently.

        Each prompt is submitted independently through the batch collector,
        which will group them into GPU-efficient batches when they overlap.

        Args:
            prompts:               List of tokenized prompts.
            max_new_tokens:        Maximum tokens to generate per prompt.
            temperature:           Sampling temperature.
            top_k:                 Top-k sampling (None = disabled).
            top_p:                 Top-p nucleus sampling (None = disabled).
            repetition_penalty:    Repetition penalty factor.

        Returns:
            List of generated token sequences, one per prompt.
        """
        results: list[list[int]] = []
        for prompt in prompts:
            result = self.infer(
                prompt_tokens=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            results.append(result)
        return results

    # -----------------------------------------------------------------
    #  Health & diagnostics
    # -----------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return a dict describing server status and diagnostics.

        Includes GPU memory stats (when available), cache stats, and the
        number of pending requests in the batch collector.
        """
        info: dict[str, Any] = {
            "running": self._running,
            "device": str(self.device),
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "model_type": type(self.model).__name__,
            "pending_requests": self.batch_collector.pending_count(),
            "cache": self.cache.stats(),
        }

        if torch.cuda.is_available():
            info["gpu"] = {
                "allocated_mb": round(
                    torch.cuda.memory_allocated(self.device) / 1024**2, 2
                ),
                "reserved_mb": round(
                    torch.cuda.memory_reserved(self.device) / 1024**2, 2
                ),
                "max_allocated_mb": round(
                    torch.cuda.max_memory_allocated(self.device) / 1024**2, 2
                ),
                "device_name": torch.cuda.get_device_name(self.device),
            }
        else:
            info["gpu"] = None

        return info

    def invalidate_cache(self, model_path: str | None = None) -> None:
        """Clear the result cache.

        Args:
            model_path: If provided, only entries whose cache key contains
                        this string are removed.  If ``None``, the entire
                        cache is cleared.
        """
        if model_path is None:
            self.cache.clear()
            print("[Server] Cache cleared (all entries)")
        else:
            # Invalidate entries matching a model path pattern.
            # Since cache keys are SHA-256 hashes, we clear the whole cache
            # when a specific model path is given (the path isn't stored in
            # the key).  This is the correct behavior: if the model changed,
            # all cached outputs are stale.
            self.cache.clear()
            print(f"[Server] Cache cleared for model_path={model_path}")
