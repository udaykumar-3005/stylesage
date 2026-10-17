# evaluation/metrics.py

"""
Metric calculation formulas for image retrieval evaluation.
These metrics help us measure how good our search system is.
"""

import numpy as np

class SearchMetrics:
    """Calculate search quality metrics for image retrieval"""
    
    @staticmethod
    def precision_at_k(retrieved_ids, relevant_ids, k=10):
        """
        Precision@K: What percentage of results are actually relevant?
        
        Args:
            retrieved_ids: List of item IDs your system returned
            relevant_ids: List of item IDs that SHOULD be returned
            k: Number of top results to consider
        
        Returns:
            Float between 0 and 1 (e.g., 0.80 = 80% precision)
        """
        top_k = retrieved_ids[:k]
        num_relevant = len(set(top_k) & set(relevant_ids))
        return num_relevant / k
    
    @staticmethod
    def recall_at_k(retrieved_ids, relevant_ids, k=10):
        """
        Recall@K: What percentage of all relevant items did we find?
        
        Args:
            retrieved_ids: List of item IDs your system returned
            relevant_ids: List of ALL relevant items in database
            k: Number of top results to consider
        
        Returns:
            Float between 0 and 1 (e.g., 0.05 = found 5% of all relevant items)
        """
        if len(relevant_ids) == 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        num_found = len(set(top_k) & set(relevant_ids))
        return num_found / len(relevant_ids)
    
    @staticmethod
    def reciprocal_rank(retrieved_ids, relevant_ids):
        """
        Reciprocal Rank: How quickly did the user find their first good result?
        
        Args:
            retrieved_ids: List of item IDs in ranking order
            relevant_ids: List of relevant item IDs
        
        Returns:
            Float between 0 and 1
            - 1.0 = first result was perfect
            - 0.5 = first good result at position 2
            - 0.1 = first good result at position 10
        """
        for position, item_id in enumerate(retrieved_ids, start=1):
            if item_id in relevant_ids:
                return 1.0 / position
        return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(all_reciprocal_ranks):
        """
        MRR: Average reciprocal rank across all test queries
        """
        return np.mean(all_reciprocal_ranks)
    
    @staticmethod
    def hit_rate_at_k(retrieved_ids, relevant_ids, k=10):
        """
        Hit Rate@K: Did we find at least ONE relevant item in top K?
        
        Returns:
            1 (success) or 0 (failure)
        """
        top_k = retrieved_ids[:k]
        has_match = any(item_id in relevant_ids for item_id in top_k)
        return 1 if has_match else 0