# Memory monitoring utilities
# Provides memory usage tracking and optimization for the Endee RAG System

import gc
import logging
import os
import psutil
import threading
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    timestamp: datetime
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    gc_objects: int
    gc_collections: Dict[int, int]


class MemoryMonitor:
    def __init__(
        self,
        warning_threshold_mb: float = 500.0,
        critical_threshold_mb: float = 1000.0,
        auto_gc_enabled: bool = True,
    ):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.auto_gc_enabled = auto_gc_enabled

        self.process = psutil.Process(os.getpid())
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        self.snapshots: list[MemorySnapshot] = []
        self.max_snapshots = 100

        self._lock = threading.Lock()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None

    # ---------------- SNAPSHOT ----------------

    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        system_memory = psutil.virtual_memory()

        gc_stats = gc.get_stats()
        gc_collections = {i: stats["collections"] for i, stats in enumerate(gc_stats)}
        gc_objects = len(gc.get_objects())

        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent=memory_percent,
            available_mb=system_memory.available / (1024 * 1024),
            gc_objects=gc_objects,
            gc_collections=gc_collections,
        )

        with self._lock:
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots.pop(0)

            if self.baseline_snapshot is None:
                self.baseline_snapshot = snapshot

        if label:
            logger.debug(
                f"[{label}] RSS={snapshot.rss_mb:.1f}MB | {snapshot.percent:.1f}%"
            )

        self._check_thresholds(snapshot)
        return snapshot

    # ---------------- THRESHOLDS ----------------

    def _check_thresholds(self, snapshot: MemorySnapshot) -> None:
        """Check memory thresholds and take appropriate action."""
        if snapshot.rss_mb >= self.critical_threshold_mb:
            logger.critical(f"CRITICAL: Memory usage {snapshot.rss_mb:.1f}MB exceeds critical threshold {self.critical_threshold_mb}MB")
            if self.auto_gc_enabled and not hasattr(self, '_gc_in_progress'):
                self._gc_in_progress = True
                try:
                    self.force_garbage_collection()
                finally:
                    delattr(self, '_gc_in_progress')
        elif snapshot.rss_mb >= self.warning_threshold_mb:
            logger.warning(f"WARNING: Memory usage {snapshot.rss_mb:.1f}MB exceeds warning threshold {self.warning_threshold_mb}MB")

    # ---------------- GC ----------------

    def _take_snapshot_no_threshold_check(self, label: str = "") -> MemorySnapshot:
        """
        Take a memory usage snapshot without triggering threshold checks.
        Used internally to avoid recursion during garbage collection.
        """
        try:
            # Get process memory info
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            # Get garbage collection info
            gc_stats = gc.get_stats()
            gc_collections = {i: stats['collections'] for i, stats in enumerate(gc_stats)}
            gc_objects = len(gc.get_objects())
            
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                rss_mb=memory_info.rss / (1024 * 1024),
                vms_mb=memory_info.vms / (1024 * 1024),
                percent=memory_percent,
                available_mb=system_memory.available / (1024 * 1024),
                gc_objects=gc_objects,
                gc_collections=gc_collections
            )
            
            # Store snapshot without threshold checking
            with self._lock:
                self.snapshots.append(snapshot)
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots.pop(0)
                
                # Set baseline if not set
                if self.baseline_snapshot is None:
                    self.baseline_snapshot = snapshot
            
            # Log snapshot if labeled
            if label:
                logger.debug(f"Memory snapshot '{label}': {snapshot.rss_mb:.1f}MB RSS, {snapshot.percent:.1f}%")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error taking memory snapshot: {str(e)}")
            # Return a minimal snapshot
            return MemorySnapshot(
                timestamp=datetime.now(),
                rss_mb=0.0, vms_mb=0.0, percent=0.0,
                available_mb=0.0, gc_objects=0, gc_collections={}
            )

    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection and return statistics.
        
        Returns:
            Dictionary with garbage collection results
        """
        try:
            logger.info("Forcing garbage collection...")
            
            # Take snapshot before GC (without threshold check to avoid recursion)
            before_snapshot = self._take_snapshot_no_threshold_check("before_gc")
            
            # Force garbage collection for all generations
            collected_objects = []
            for generation in range(3):
                collected = gc.collect(generation)
                collected_objects.append(collected)
            
            # Take snapshot after GC (without threshold check)
            after_snapshot = self._take_snapshot_no_threshold_check("after_gc")
            
            # Calculate memory freed
            memory_freed_mb = before_snapshot.rss_mb - after_snapshot.rss_mb
            objects_freed = before_snapshot.gc_objects - after_snapshot.gc_objects
            
            gc_results = {
                "memory_freed_mb": memory_freed_mb,
                "objects_freed": objects_freed,
                "collected_by_generation": collected_objects,
                "total_collected": sum(collected_objects),
                "before_memory_mb": before_snapshot.rss_mb,
                "after_memory_mb": after_snapshot.rss_mb
            }
            
            logger.info(f"Garbage collection completed: freed {memory_freed_mb:.1f}MB, {objects_freed} objects")
            return gc_results
            
        except Exception as e:
            logger.error(f"Error during garbage collection: {str(e)}")
            return {"error": str(e)}

    # ---------------- STATS ----------------

    def get_memory_stats(self) -> Dict[str, Any]:
        current = self.take_snapshot("stats")

        stats = {
            "current": {
                "rss_mb": current.rss_mb,
                "vms_mb": current.vms_mb,
                "percent": current.percent,
                "available_mb": current.available_mb,
                "gc_objects": current.gc_objects,
            },
            "thresholds": {
                "warning_mb": self.warning_threshold_mb,
                "critical_mb": self.critical_threshold_mb,
                "auto_gc_enabled": self.auto_gc_enabled,
            },
            "snapshots_count": len(self.snapshots),
        }

        if self.baseline_snapshot:
            stats["growth_since_baseline"] = {
                "rss_mb": current.rss_mb - self.baseline_snapshot.rss_mb,
                "vms_mb": current.vms_mb - self.baseline_snapshot.vms_mb,
                "gc_objects": current.gc_objects - self.baseline_snapshot.gc_objects,
                "time_elapsed": (
                    current.timestamp - self.baseline_snapshot.timestamp
                ).total_seconds(),
            }

        return stats

    # ---------------- OPTIMIZATION ----------------

    def optimize_memory(self) -> Dict[str, Any]:
        """
        Perform comprehensive memory optimization with enhanced features.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        MAX_SECONDS = 5.0  # Increased timeout for more thorough optimization

        results = {
            "started_at": datetime.now().isoformat(),
            "actions_taken": [],
            "memory_freed_mb": 0.0,
            "errors": [],
        }

        try:
            logger.info("Starting comprehensive memory optimization...")
            
            # Take initial snapshot (without threshold check to avoid recursion)
            initial = self._take_snapshot_no_threshold_check("opt_start")
            results["initial_memory_mb"] = initial.rss_mb

            # Action 1: Force garbage collection
            try:
                results["gc_results"] = self.force_garbage_collection()
                results["actions_taken"].append("garbage_collection")
            except Exception as e:
                error_msg = f"Garbage collection failed: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"Memory optimization - {error_msg}", exc_info=True)

            # Action 2: Optimize embedding cache
            try:
                from embedding_cache import embedding_cache
                cache = embedding_cache.optimize_memory()
                results["cache_optimization"] = cache
                results["actions_taken"].append("embedding_cache_optimization")
            except Exception as e:
                error_msg = f"Cache optimization failed: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"Memory optimization - {error_msg}", exc_info=True)

            # Action 3: Clear old snapshots to free memory
            try:
                with self._lock:
                    if len(self.snapshots) > 10:  # Keep only last 10 snapshots
                        old_count = len(self.snapshots)
                        self.snapshots = self.snapshots[-10:]
                        results["actions_taken"].append("snapshot_cleanup")
                        results["snapshots_cleared"] = old_count - len(self.snapshots)
                        logger.debug(f"Cleared {old_count - len(self.snapshots)} old memory snapshots")
            except Exception as e:
                error_msg = f"Snapshot cleanup failed: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"Memory optimization - {error_msg}", exc_info=True)

            # Action 4: System-level memory optimization
            try:
                # Clear Python's internal caches
                import importlib
                if hasattr(importlib, 'invalidate_caches'):
                    importlib.invalidate_caches()
                    results["actions_taken"].append("import_cache_clear")
                    logger.debug("Cleared Python import caches")
                
                # Additional memory optimization
                import sys
                # Force garbage collection of modules that are no longer needed
                modules_before = len(sys.modules)
                # Note: We don't actually remove modules as it could break the application
                results["modules_count"] = modules_before
                results["actions_taken"].append("system_optimization")
                
            except Exception as e:
                error_msg = f"System optimization failed: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"Memory optimization - {error_msg}", exc_info=True)

            # Check timeout
            if time.time() - start_time > MAX_SECONDS:
                results["errors"].append(f"Optimization timed out after {MAX_SECONDS} seconds")

            # Take final snapshot
            final = self._take_snapshot_no_threshold_check("opt_end")
            results["final_memory_mb"] = final.rss_mb
            results["memory_freed_mb"] = initial.rss_mb - final.rss_mb
            results["completed_at"] = datetime.now().isoformat()
            results["duration_seconds"] = time.time() - start_time

            logger.info(f"Memory optimization completed: freed {results['memory_freed_mb']:.1f}MB in {results['duration_seconds']:.2f}s")

        except Exception as e:
            results["errors"].append(f"Optimization failed: {str(e)}")
            logger.error(f"Memory optimization error: {str(e)}")

        return results

    # ---------------- MONITOR THREAD ----------------

    def start_monitoring(self, interval_seconds: float = 30.0) -> None:
        if self._monitoring_active:
            return

        self._monitoring_active = True

        def loop():
            while self._monitoring_active:
                self.take_snapshot("monitor")
                time.sleep(interval_seconds)

        self._monitor_thread = threading.Thread(target=loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

    # ---------------- BASELINE ----------------

    def reset_baseline(self) -> None:
        snapshot = self.take_snapshot("baseline_reset")
        with self._lock:
            self.baseline_snapshot = snapshot

    # ---------------- RECOMMENDATIONS ----------------

    def get_memory_recommendations(self) -> list[str]:
        stats = self.get_memory_stats()
        current_mb = stats["current"]["rss_mb"]
        recs = []

        if current_mb > self.critical_threshold_mb:
            recs.append("Critical memory usage detected.")
        elif current_mb > self.warning_threshold_mb:
            recs.append("Memory usage elevated.")

        if not recs:
            recs.append("Memory usage is within normal range.")

        return recs


# Global instance
memory_monitor = MemoryMonitor()
