
import pytest
from unittest.mock import patch, mock_open
from seahorse.resource_manager import ResourceManager

@pytest.fixture
def mock_psutil():
    with patch("psutil.virtual_memory") as mock_vm, \
         patch("psutil.swap_memory") as mock_sm:
        yield mock_vm, mock_sm

def test_calculate_node_budget_basic():
    rm = ResourceManager()
    # M_effective = 1000MB, alpha = 0.1, C_node = 2KB
    # Budget = (1000 * 1024 * 0.1) / 2 = 51200 / 2 = 25600
    # But N_max might limit it.
    
    rm.profile_params = {"alpha": 0.1, "n_max": 5000}
    budget = rm.calculate_node_budget(available_mb=1000)
    assert budget == 5000 # Limited by n_max

    rm.profile_params = {"alpha": 0.1, "n_max": 100000}
    budget = rm.calculate_node_budget(available_mb=1000)
    # 1000 * 1024 * 0.1 / 2 = 51200
    assert budget == 51200


def test_get_available_memory_no_zram(mock_psutil):
    mock_vm, mock_sm = mock_psutil
    mock_vm.return_value.available = 1024 * 1024 * 500 # 500MB
    
    rm = ResourceManager()
    with patch("builtins.open", side_effect=FileNotFoundError):
        mem = rm.get_available_memory()
        assert mem == 500 # Only physical available

def test_get_available_memory_with_zram(mock_psutil):
    mock_vm, mock_sm = mock_psutil
    mock_vm.return_value.available = 1024 * 1024 * 500 # 500MB
    mock_sm.return_value.free = 1024 * 1024 * 1000 # 1000MB swap free
    
    # Mock /proc/swaps to show zram
    swaps_content = "Filename\tType\tSize\tUsed\tPriority\n/dev/zram0\tpartition\t2097148\t0\t5\n"
    
    rm = ResourceManager()
    with patch("builtins.open", mock_open(read_data=swaps_content)):
        mem = rm.get_available_memory()
        # 500MB + (1000MB * 0.5) = 1000MB
        assert mem == 1000

def test_profile_selection():
    rm = ResourceManager()
    
    # Low Spec
    params = rm._select_profile(total_mb=2000)
    assert params["n_max"] == 2000
    assert params["alpha"] == 0.05
    
    # Mid Spec
    params = rm._select_profile(total_mb=8000)
    assert params["n_max"] == 5000
    assert params["alpha"] == 0.10
    
    # High Spec
    params = rm._select_profile(total_mb=16000)
    assert params["n_max"] == 10000
    assert params["alpha"] == 0.20
