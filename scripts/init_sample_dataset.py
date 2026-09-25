"""Populate initial dataset/ directory with sample records matching official format."""

import os
import pandas as pd

def init_sample_dataset():
    data_dir = os.path.abspath("dataset")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Train S1 (US and India)
    train_s1 = pd.DataFrame([
        {"entity_id": "S1-00001", "business_name": "Apex Technology Solutions Inc.", "business_address": "100 Market St, Suite 400, San Francisco, CA 94105", "country": "US"},
        {"entity_id": "S1-00002", "business_name": "Sharma Textiles Pvt Ltd", "business_address": "45 Ring Road, Near Surat Railway Station, Surat 395002", "country": "India"},
        {"entity_id": "S1-00003", "business_name": "Lone Star Singleton Diner", "business_address": "99 Desert Hwy, Austin, TX 78701", "country": "US"},
        {"entity_id": "S1-00004", "business_name": "Global Logistics Corp dba Speedy Freight", "business_address": "12 Harbor Blvd, Boston, MA 02110", "country": "US"},
        {"entity_id": "S1-00005", "business_name": "Patel Sweets & Snacks", "business_address": "Opposite Town Hall, MG Road, Ahmedabad 380001", "country": "India"},
        {"entity_id": "S1-00006", "business_name": "Bay Area Micro Brewery LLC", "business_address": "742 Evergreen Terr, San Jose, CA 95112", "country": "US"},
        {"entity_id": "S1-00007", "business_name": "Kavitha Silks & Sarees", "business_address": "12 Gandhi Bazaar, Basavanagudi, Bangalore 560004", "country": "India"},
    ])

    # Train S2
    train_s2 = pd.DataFrame([
        {"entity_id": "S2-00001", "business_name": "Apex Tech Solutions", "business_address": "100 Market Street, Ste 400, SF, 94105", "country": "US"},
        {"entity_id": "S2-00002", "business_name": "Sharma Textiles", "business_address": "45 Ring Rd, Nr Rly Station, Surat, Gujarat 395002", "country": "India"},
        {"entity_id": "S2-00003", "business_name": "Speedy Freight", "business_address": "12 Harbor Boulevard, Boston 02110", "country": "US"},
        {"entity_id": "S2-00004", "business_name": "Unrelated Chicago Pizza Co", "business_address": "22 Michigan Ave, Chicago, IL 60601", "country": "US"},
        {"entity_id": "S2-00005", "business_name": "Bay Area Micro Brew", "business_address": "742 Evergreen Terrace, San Jose 95112", "country": "US"},
        {"entity_id": "S2-00006", "business_name": "Kavitha Silks", "business_address": "Gandhi Bazar, Bangalore 560004", "country": "India"},
    ])

    # Train S3
    train_s3 = pd.DataFrame([
        {"entity_id": "S3-00001", "business_name": "Apex Technology Inc", "business_address": "100 Market St, San Francisco 94105", "country": "US"},
        {"entity_id": "S3-00002", "business_name": "Patel Sweets and Snacks", "business_address": "Opp Town Hall, M.G. Road, Ahmedabad 380001", "country": "India"},
        {"entity_id": "S3-00003", "business_name": "Random Indian Bakery", "business_address": "78 Brigade Rd, Bangalore 560025", "country": "India"},
        {"entity_id": "S3-00004", "business_name": "Evergreen Microbrewery", "business_address": "742 Evergreen Ter, San Jose, CA 95112", "country": "US"},
    ])

    # Train Ground Truth
    train_gt = pd.DataFrame([
        {"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00001,S3-00001"},
        {"source1_entity_id": "S1-00002", "matched_entity_ids": "S2-00002"},
        {"source1_entity_id": "S1-00003", "matched_entity_ids": ""}, # Singleton
        {"source1_entity_id": "S1-00004", "matched_entity_ids": "S2-00003"},
        {"source1_entity_id": "S1-00005", "matched_entity_ids": "S3-00002"},
        {"source1_entity_id": "S1-00006", "matched_entity_ids": "S2-00005,S3-00004"},
        {"source1_entity_id": "S1-00007", "matched_entity_ids": "S2-00006"},
    ])

    # Test S1 (US, India, and France)
    test_s1 = pd.DataFrame([
        {"entity_id": "S1-00001", "business_name": "Dubois Boulangerie S.A.R.L.", "business_address": "14 Bd. Saint-Germain, 75005 Paris", "country": "France"},
        {"entity_id": "S1-00002", "business_name": "Acme Widgets LLC", "business_address": "500 Broadway, New York, NY 10012", "country": "US"},
        {"entity_id": "S1-00003", "business_name": "Bengaluru IT Consultants", "business_address": "100 Outer Ring Rd, Marathahalli, Bangalore 560037", "country": "India"},
        {"entity_id": "S1-00004", "business_name": "Parisian Gourmet Cafe E.U.R.L.", "business_address": "8 Rue Montorgueil, 75001 Paris", "country": "France"},
        {"entity_id": "S1-00005", "business_name": "Unmatched Texas Sole Proprietor", "business_address": "404 Nowhere Rd, El Paso, TX 79901", "country": "US"},
    ])

    # Test S2
    test_s2 = pd.DataFrame([
        {"entity_id": "S2-00001", "business_name": "Boulangerie Dubois", "business_address": "14 Boulevard Saint Germain, 75005 Paris", "country": "France"},
        {"entity_id": "S2-00002", "business_name": "Acme Widgets", "business_address": "500 Broadway St, NY 10012", "country": "US"},
        {"entity_id": "S2-00003", "business_name": "Cafe Paris Gourmet", "business_address": "8 Rue Montorgueil, Paris 75001", "country": "France"},
        {"entity_id": "S2-00004", "business_name": "Completely Unrelated Shop", "business_address": "10 Rue de la Paix, Paris 75002", "country": "France"},
    ])

    # Test S3
    test_s3 = pd.DataFrame([
        {"entity_id": "S3-00001", "business_name": "Dubois Boulangerie SARL", "business_address": "14 Bd St Germain, Paris 75005", "country": "France"},
        {"entity_id": "S3-00002", "business_name": "Bangalore IT Consulting Pvt Ltd", "business_address": "100 Outer Ring Road, Bangalore 560037", "country": "India"},
        {"entity_id": "S3-00003", "business_name": "Acme Widgets Corp", "business_address": "500 Broadway, Suite 2, New York, NY 10012", "country": "US"},
    ])

    train_s1.to_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", index=False)
    train_s2.to_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", index=False)
    train_s3.to_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", index=False)
    train_gt.to_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    test_s1.to_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", index=False)
    test_s2.to_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", index=False)
    test_s3.to_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", index=False)
    print("Sample dataset created successfully.")

if __name__ == "__main__":
    init_sample_dataset()
