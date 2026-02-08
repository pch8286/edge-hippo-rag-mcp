
import psutil
import logging
import os
from typing import List, Dict, Optional
from .config import settings

logger = logging.getLogger(__name__)

class ResourceManager:
    _instance = None
    
    PROFILES = {
        "low": {"alpha": 0.05, "n_max": 2000},
        "mid": {"alpha": 0.10, "n_max": 5000},
        "high": {"alpha": 0.20, "n_max": 10000},
        "max": {"alpha": 1.0, "n_max": 1000000}, # Uncapped (Infinite) budget
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.profile = settings.HIPPO_PERFORMANCE_PROFILE
        self.profile_params = self._initialize_profile()
        self._initialized = True

    def _initialize_profile(self) -> Dict[str, float]:
        """Determine profile parameters based on config or auto-scaling."""
        if self.profile == "auto":
            total_mb = psutil.virtual_memory().total / (1024 * 1024)
            params = self._select_profile(total_mb)
            logger.info(f"Auto-scaling: Detected {total_mb:.0f}MB RAM. Selecting '{self.profile}' profile -> {params}")
        elif self.profile in self.PROFILES:
            params = self.PROFILES[self.profile].copy()
            logger.info(f"Using manual profile: {self.profile}")
        else:
            logger.warning(f"Unknown profile '{self.profile}'. Falling back to 'mid'.")
            params = self.PROFILES["mid"].copy()

        if settings.HIPPO_NODE_MAX:
            params["n_max"] = settings.HIPPO_NODE_MAX
        if settings.HIPPO_MEMORY_ALPHA:
            params["alpha"] = settings.HIPPO_MEMORY_ALPHA
            
        return params

    def _select_profile(self, total_mb: float) -> Dict[str, float]:
        """Internal heuristic for auto-scaling."""
        if total_mb < 4000:
            return self.PROFILES["low"].copy()
        elif total_mb < 8500: # RPi 5 8GB is ~8100MB
            return self.PROFILES["mid"].copy()
        elif total_mb < 16500: # Standard 16GB
            return self.PROFILES["high"].copy()
        else: # High-end hardware (>16GB)
            return self.PROFILES["max"].copy()

    def _is_zram_swap(self) -> bool:
        """Check if any swap device is zram."""
        try:
            if not os.path.exists("/proc/swaps"):
                return False
            with open("/proc/swaps", "r") as f:
                content = f.read()
                return "zram" in content.lower()
        except Exception:
            return False

    def get_available_memory(self) -> float:
        """Get effective available memory in MB."""
        vm = psutil.virtual_memory()
        available_mb = vm.available / (1024 * 1024)
        
        if self._is_zram_swap():
            sm = psutil.swap_memory()
            zram_free_mb = sm.free / (1024 * 1024)
            # Add 50% of free zram to effective pool
            effective_mb = available_mb + (zram_free_mb * 0.5)
            logger.debug(f"ZRAM detected. Effective Memory: {effective_mb:.1f}MB (Phys: {available_mb:.1f}MB, ZRAM_Free: {zram_free_mb:.1f}MB)")
            return effective_mb
            
        return available_mb

    def calculate_node_budget(self, available_mb: Optional[float] = None) -> int:
        """
        Calculate max nodes allowed for current query.
        Formula: N = min(n_max, (M_available * alpha) / C_node)
        Where C_node is estimated at 2KB/node.
        """
        if available_mb is None:
            available_mb = self.get_available_memory()
            
        alpha = self.profile_params["alpha"]
        n_max = self.profile_params["n_max"]
        budget = int((available_mb * 1024 * alpha) / 2)
        final_limit = min(n_max, budget)
        final_limit = max(500, final_limit)
        
        logger.debug(f"Calculated Node Budget: {final_limit} (Available: {available_mb:.1f}MB, Alpha: {alpha})")
        return final_limit

resource_manager = ResourceManager()
