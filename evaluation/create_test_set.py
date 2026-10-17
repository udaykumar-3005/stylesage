# evaluation/create_test_set.py

"""
Create a diverse test set by sampling from your dataset.
This generates test_queries.json automatically.
"""

import pickle
import pandas as pd
import json
import random

# 🔧 UPDATED: Use your actual filename
print("📂 Loading metadata from artifacts/clean_metadata.pkl...")
with open('artifacts/clean_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

df = pd.DataFrame(metadata)

print(f"✅ Loaded {len(df):,} items")
print(f"Columns: {list(df.columns)}")

def create_test_set(df, output_file='evaluation/test_queries.json'):
    """
    Create a diverse test set by sampling category-color combinations
    """
    
    # Define test scenarios (adjust based on your analyze_dataset.py output)
    # Format: (category, color, number_of_queries)
    test_scenarios = [
        ('Tshirts', 'Blue', 2),
        ('Tshirts', 'Black', 2),
        ('Shirts', 'Blue', 2),
        ('Shirts', 'White', 2),
        ('Jeans', 'Blue', 1),
        ('Casual Shoes', 'Black', 1),
        ('Sports Shoes', 'White', 1),
        ('Watches', 'Black', 1),
        ('Flip Flops', 'Black', 1),
        ('Handbags', 'Black', 1),
        # Add more based on your dataset
    ]
    
    test_queries = []
    query_id = 1
    
    print("\n" + "="*70)
    print("🔍 CREATING TEST SET")
    print("="*70)
    
    for category, color, n_queries in test_scenarios:
        # Find all items matching this category + color
        matching_items = df[
            (df['articleType'] == category) & 
            (df['baseColour'] == color)
        ]
        
        if len(matching_items) < 5:
            print(f"⚠️  Skipping: {category} + {color} (only {len(matching_items)} items)")
            continue
        
        # Randomly sample ONE item as the query image
        query_item = matching_items.sample(n=1).iloc[0]
        
        # All OTHER matching items are the ground truth
        relevant_items = matching_items[matching_items['id'] != query_item['id']]
        relevant_ids = relevant_items['id'].tolist()
        
        # Create test case
        test_case = {
            'query_id': query_id,
            'query_item_id': int(query_item['id']),
            'query_image': query_item['image_path'],
            'description': query_item.get('productDisplayName', f"{color} {category}"),
            'category': category,
            'color': color,
            'relevant_ids': [int(x) for x in relevant_ids],  # Convert to int
            'num_relevant': len(relevant_ids)
        }
        
        test_queries.append(test_case)
        
        print(f"✅ Query {query_id}: {category} + {color}")
        print(f"   Query Image ID: {query_item['id']}")
        print(f"   Ground Truth: {len(relevant_ids)} similar items")
        
        query_id += 1
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ Created {len(test_queries)} test queries")
    print(f"📁 Saved to: {output_file}")
    print("="*70)
    
    return test_queries

# Run it
if __name__ == "__main__":
    test_queries = create_test_set(df)