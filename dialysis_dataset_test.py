import unittest
import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import dataset classes
from dialysis_dataset import DialysisDataset, DialysisDataModule

class TestDialysisDataset(unittest.TestCase):
    """Test cases for Dialysis Dataset and DataModule classes."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources before any tests run."""
        cls.test_data_path = '/tmp/test_dialysis_data.pkl'
        cls.prepare_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        if os.path.exists(cls.test_data_path):
            try:
                os.remove(cls.test_data_path)
                print(f"Removed temporary test data: {cls.test_data_path}")
            except Exception as e:
                print(f"Could not remove test file: {e}")
    
    @classmethod
    def prepare_test_data(cls):
        """Prepare test data with dummy target."""
        data_path = '/home/jovyan/work/Data/LevanteDP/pickle_df/analiticas.pkl'
        
        # Check if the data file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        print(f"Loading data from {data_path}")
        df = pd.read_pickle(data_path)
        print(f"Original dataframe shape: {df.shape}")
        
        # Create a dummy target column if it doesn't exist
        if 'outcome' not in df.columns:
            # Use albumin < median as a dummy target for testing
            if 'ALBUMINA' in df.columns:
                median_albumin = df['ALBUMINA'].median()
                df['outcome'] = (df['ALBUMINA'] < median_albumin).astype(int)
                print(f"Created dummy 'outcome' target using ALBUMINA < {median_albumin}")
            else:
                # Random outcome if ALBUMINA not available
                df['outcome'] = np.random.randint(0, 2, size=len(df))
                print("Created random dummy 'outcome' target")
        
        # Save the temporary dataframe for testing
        df.to_pickle(cls.test_data_path)
        print(f"Test data saved to {cls.test_data_path}")
    
    def test_a_dataset_creation(self):
        """Test creating and setting up DialysisDataset objects."""
        # Create dataset for each split
        train_dataset = DialysisDataset('train', data_path=self.test_data_path, n_timesteps=32)
        val_dataset = DialysisDataset('val', data_path=self.test_data_path, n_timesteps=32)
        test_dataset = DialysisDataset('test', data_path=self.test_data_path, n_timesteps=32)
        
        # Setup datasets
        train_dataset.setup()
        val_dataset.setup()
        test_dataset.setup()
        
        # Check that each split has at least one sample
        self.assertGreater(len(train_dataset), 0, "Train dataset has no samples")
        self.assertGreater(len(val_dataset), 0, "Validation dataset has no samples")
        self.assertGreater(len(test_dataset), 0, "Test dataset has no samples")
        
        print(f"Train set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
    
    def test_b_getitem_function(self):
        """Test the __getitem__ function of DialysisDataset."""
        train_dataset = DialysisDataset('train', data_path=self.test_data_path, n_timesteps=32)
        train_dataset.setup()
        
        # Get a single item
        x, y = train_dataset[0]
        
        # Unpack the tuple
        x_ts, x_static, bin_ends = x
        
        print(f"Time series tensor shape: {x_ts.shape}")
        print(f"Static features tensor shape: {x_static.shape}")
        print(f"Bin ends tensor shape: {bin_ends.shape}")
        print(f"Target value: {y}")
        
        # Check expected shapes
        expected_ts_shape = (32, train_dataset.d_time_series_num()*2)
        self.assertEqual(x_ts.shape, expected_ts_shape, 
                         f"Time series shape {x_ts.shape} doesn't match expected {expected_ts_shape}")
        
        expected_static_shape = (train_dataset.d_static_num(),)
        self.assertEqual(x_static.shape, expected_static_shape, 
                         f"Static features shape {x_static.shape} doesn't match expected {expected_static_shape}")
            
        self.assertEqual(bin_ends.shape, (32,), 
                         f"Bin ends shape {bin_ends.shape} doesn't match expected (32,)")
        
        # Check that tensors have the right type
        self.assertIsInstance(x_ts, torch.Tensor, "Time series should be a torch.Tensor")
        self.assertIsInstance(x_static, torch.Tensor, "Static features should be a torch.Tensor")
        self.assertIsInstance(bin_ends, torch.Tensor, "Bin ends should be a torch.Tensor")
    
    def test_c_dataloader_iteration(self):
        """Test that we can iterate over the data loader."""
        batch_size = 4
        
        # Create data module
        dm = DialysisDataModule(
            data_path=self.test_data_path,
            batch_size=batch_size,
            num_workers=0  # Use 0 for easier debugging
        )
        
        dm.setup()
        
        # Get the train dataloader
        train_dl = dm.train_dataloader()
        print(f"DataLoader created with batch size {batch_size}")
        
        # Iterate over a few batches
        batch_count = 0
        for i, (x_batch, y_batch) in enumerate(train_dl):
            if i >= 2:  # Test just the first 2 batches
                break
                
            # Unpack batch data
            x_ts_batch, x_static_batch, bin_ends_batch = x_batch
            
            print(f"\nBatch {i+1}:")
            print(f"- Time series batch shape: {x_ts_batch.shape}")
            print(f"- Static features batch shape: {x_static_batch.shape}")
            print(f"- Bin ends batch shape: {bin_ends_batch.shape}")
            print(f"- Labels shape: {y_batch.shape}")
            
            # Verify batch size (might be smaller for last batch)
            max_expected_batch_size = batch_size
            self.assertLessEqual(x_ts_batch.shape[0], max_expected_batch_size, 
                                "Batch size exceeds specified maximum")
            
            batch_count += 1
        
        self.assertGreater(batch_count, 0, "No batches yielded from DataLoader")
    
    def test_d_data_module(self):
        """Test the DialysisDataModule class."""
        # Create the data module
        dm = DialysisDataModule(
            data_path=self.test_data_path,
            batch_size=8,
            num_workers=0
        )
        
        # Setup
        print("Setting up DataModule...")
        dm.setup()
        
        # Check dimensions and target info
        print(f"Time series feature dimension: {dm.d_time_series_num()}")
        print(f"Static feature dimension: {dm.d_static_num()}")
        print(f"Target dimension: {dm.d_target()}")
        print(f"Positive class fraction: {dm.pos_frac()}")
        
        # Assert valid dimensions
        self.assertGreater(dm.d_time_series_num(), 0, "Time series feature dimension should be positive")
        self.assertGreater(dm.d_static_num(), 0, "Static feature dimension should be positive")
        self.assertGreaterEqual(dm.d_target(), 1, "Target dimension should be at least 1")
        self.assertGreaterEqual(dm.pos_frac(), 0.0, "Positive class fraction should be non-negative")
        self.assertLessEqual(dm.pos_frac(), 1.0, "Positive class fraction should be at most 1")
        
        # Get dataloaders
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()
        
        self.assertIsNotNone(train_dl, "Train DataLoader is None")
        self.assertIsNotNone(val_dl, "Validation DataLoader is None")
        self.assertIsNotNone(test_dl, "Test DataLoader is None")
        
        print("DataLoaders created successfully")
    
    def test_e_visualization(self):
        """Test patient data visualization capability."""
        train_dataset = DialysisDataset('train', data_path=self.test_data_path, n_timesteps=32)
        train_dataset.setup()
        
        if len(train_dataset) == 0:
            self.skipTest("No patients available for visualization")
            
        patient_idx = 0
        print(f"Visualizing patient {patient_idx} data")
        
        # Get the patient data
        x, y = train_dataset[patient_idx]
        x_ts, x_static, bin_ends = x
        
        # Get only the value part (not the mask)
        n_features = train_dataset.d_time_series_num()
        values = x_ts[:, :n_features].numpy()
        mask = x_ts[:, n_features:].numpy()
        
        # Create figure
        fig, axes = plt.subplots(4, 5, figsize=(20, 15))
        axes = axes.flatten()
        
        # Plot each lab value across time bins
        for i in range(min(n_features, 18)):
            ax = axes[i]
            # Only plot points where the mask is non-zero
            valid_bins = mask[:, i] > 0
            valid_times = np.arange(len(bin_ends))[valid_bins]
            valid_values = values[valid_bins, i]
            
            if len(valid_times) > 0:
                ax.scatter(valid_times, valid_values, color='blue')
                ax.plot(valid_times, valid_values, 'b-', alpha=0.5)
                ax.set_title(train_dataset.lab_columns[i])
                ax.set_xlabel('Time bin')
                ax.set_ylabel('Normalized value')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(train_dataset.lab_columns[i])
        
        plt.tight_layout()
        output_path = '/tmp/patient_visualization.png'
        plt.savefig(output_path)
        plt.close(fig)
        
        self.assertTrue(os.path.exists(output_path), 
                        f"Visualization file was not created at {output_path}")
        print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    unittest.main()