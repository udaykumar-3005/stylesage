# evaluation/evaluator.py

"""
Evaluate the search system using the test set.
Calculates Precision, Recall, MRR, and Latency.
"""

import json
import time
import numpy as np
from evaluation.metrics import SearchMetrics
from src.retriever.search_engine import SearchEngine 


class ImageSearchEvaluator:
    def __init__(self):
        print("🔧 Loading search engine...")
        self.search_engine = SearchEngine()  # 🔧 Update this to match your actual class
        self.metrics = SearchMetrics()
        print("✅ Search engine loaded")
    
    def evaluate(self, test_file='evaluation/test_queries.json', k=10):
        """
        Run evaluation on all test images
        """
        # Load test cases
        with open(test_file) as f:
            test_queries = json.load(f)
        
        # Storage for metric values
        results = {
            'precision': [],
            'recall': [],
            'reciprocal_ranks': [],
            'hit_rates': [],
            'latencies': []
        }
        
        print("\n" + "="*70)
        print(f"🧪 EVALUATING ON {len(test_queries)} TEST QUERIES")
        print("="*70)
        
        # Process each test query
        for query in test_queries:
            image_path = query['query_image']
            query_item_id = query['query_item_id']
            relevant_ids = query['relevant_ids']
            description = query['description']
            
            print(f"\n[Query {query['query_id']}] {description}")
            print(f"📷 Query Image ID: {query_item_id}")
            print(f"✅ Ground Truth: {len(relevant_ids)} similar items")
            
            # Run search and measure time
            start_time = time.time()
            
            # 🔧 UPDATE THIS LINE to match your search function
            search_results = self.search_engine.search_by_image(
                image_path,
                top_k=k+1  # Get k+1 to account for removing self-match
            )
            
            latency = (time.time() - start_time) * 1000  # ms
            
            # Extract IDs from results
            # 🔧 UPDATE THIS based on your result format
            retrieved_ids = [result['id'] for result in search_results]
            
            # Remove the query image itself (self-match)
            if query_item_id in retrieved_ids:
                retrieved_ids.remove(query_item_id)
                print(f"   ℹ️  Removed self-match (ID {query_item_id})")
            
            # Take only top K after removing self-match
            retrieved_ids = retrieved_ids[:k]
            
            # Calculate metrics
            precision = self.metrics.precision_at_k(retrieved_ids, relevant_ids, k)
            recall = self.metrics.recall_at_k(retrieved_ids, relevant_ids, k)
            rr = self.metrics.reciprocal_rank(retrieved_ids, relevant_ids)
            hit = self.metrics.hit_rate_at_k(retrieved_ids, relevant_ids, k)
            
            # Store
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['reciprocal_ranks'].append(rr)
            results['hit_rates'].append(hit)
            results['latencies'].append(latency)
            
            # Print individual results
            num_relevant_found = int(precision * k)
            print(f"   Precision@{k}: {precision:.3f} ({num_relevant_found}/{k} relevant)")
            print(f"   Recall@{k}: {recall:.3f}")
            print(f"   RR: {rr:.3f}")
            print(f"   Latency: {latency:.0f}ms")
        
        # Calculate averages
        avg_metrics = {
            f'Precision@{k}': np.mean(results['precision']),
            f'Recall@{k}': np.mean(results['recall']),
            'MRR': np.mean(results['reciprocal_ranks']),
            f'Hit Rate@{k}': np.mean(results['hit_rates']),
            'Avg Latency (ms)': np.mean(results['latencies']),
            'P95 Latency (ms)': np.percentile(results['latencies'], 95)
        }
        
        # Print summary
        print("\n" + "="*70)
        print("📊 FINAL EVALUATION RESULTS")
        print("="*70)
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name:.<50} {value:.3f}")
        print("="*70)
        
        return avg_metrics

if __name__ == "__main__":
    evaluator = ImageSearchEvaluator()
    results = evaluator.evaluate(k=10)
    
    # Save results
    with open('evaluation/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to evaluation/results.json")