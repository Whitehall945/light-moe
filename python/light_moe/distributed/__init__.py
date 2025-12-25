"""
Distributed communication utilities.
"""

from light_moe.distributed.comm_group import CommGroup
from light_moe.distributed.expert_parallel import ExpertParallelGroup

__all__ = ["CommGroup", "ExpertParallelGroup"]
