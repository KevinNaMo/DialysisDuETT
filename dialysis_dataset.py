import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Manager
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

class DialysisDataset(Dataset):
    def __init__(self, split_name, data_path=None, n_timesteps=32, use_temp_cache=False, 
                 target_col='outcome', seed=0, **kwargs):
        """
        Dataset for dialysis lab data
        
        Args:
            split_name: 'train', 'val', or 'test'
            data_path: path to the pickle file with the analytics data
            n_timesteps: Number of time bins to use for discretization
            use_temp_cache: Whether to use a temporary cache for processed items
            target_col: Column name for the target variable
            seed: Random seed for train/val/test splits
        """
        self.split_name = split_name
        self.n_timesteps = n_timesteps
        self.temp_cache = Manager().dict() if use_temp_cache else None
        self.data_path = data_path
        self.target_col = target_col
        self.seed = seed
        
        # Lab values to use
        self.lab_columns = [
            'GLUCOSA', 'UREA', 'CREATININA', 'URICO', 'SODIO', 'POTASIO', 
            'CALCIO', 'FOSFORO', 'HIERRO', 'TRANSFERRINA', 'IST', 'FERRITINA', 
            'COLESTEROL', 'TRIGLICERIDOS', 'HDL', 'LDL', 'PROTEINAS', 'ALBUMINA'
        ]
        
        # Will be populated in setup()
        self.X = None
        self.y = None
        self.patient_ids = None
        self.means = []
        self.stds = []
        self.maxes = []
        self.mins = []

    def setup(self):
        """Load and preprocess the data"""
        # Load analytics data
        analiticas = pd.read_pickle(self.data_path)
        
        # Convert date to datetime if not already
        analiticas['FECHA'] = pd.to_datetime(analiticas['FECHA'])
        
        # Sort by patient ID and date
        analiticas = analiticas.sort_values(['REGISTRO', 'FECHA'])
        
        # For this example, we'll use a simple dummy target - you should replace this
        # with your actual target variable
        if self.target_col not in analiticas.columns:
            print(f"Warning: Target column {self.target_col} not found. Creating dummy target.")
            # Create a dummy binary outcome (replace with actual outcome)
            analiticas['outcome'] = (analiticas['ALBUMINA'] < analiticas['ALBUMINA'].median()).astype(int)
        
        # Get unique patients
        patient_ids = analiticas['REGISTRO'].unique()
        
        # Create train/val/test splits by patient ID to avoid data leakage
        train_ids, test_val_ids = train_test_split(
            patient_ids, test_size=0.3, random_state=self.seed
        )
        val_ids, test_ids = train_test_split(
            test_val_ids, test_size=0.5, random_state=self.seed
        )
        
        # Select the appropriate patient IDs based on split
        if self.split_name == 'train':
            selected_ids = train_ids
        elif self.split_name == 'val':
            selected_ids = val_ids
        elif self.split_name == 'test':
            selected_ids = test_ids
        else:
            raise ValueError(f"Invalid split name: {self.split_name}")
            
        # Filter data for selected patients
        split_data = analiticas[analiticas['REGISTRO'].isin(selected_ids)]
        
        # Create patient-level data structures
        patients_data = []
        patients_outcomes = []
        
        for patient_id in selected_ids:
            patient_df = analiticas[analiticas['REGISTRO'] == patient_id]
            
            # Skip patients with less than 2 measurements
            if len(patient_df) < 2:
                continue
                
            # Get lab values
            labs = patient_df[self.lab_columns].values
            
            # Calculate time in days from first measurement
            start_date = patient_df['FECHA'].min()
            days_since_start = (patient_df['FECHA'] - start_date).dt.total_seconds() / (24 * 3600)
            
            # Stack time and labs
            patient_data = np.column_stack([days_since_start.values, labs])
            
            # Get outcome (use the most recent outcome for the patient)
            outcome = patient_df[self.target_col].iloc[-1]
            
            patients_data.append(patient_data)
            patients_outcomes.append(outcome)
        
        # Convert to tensors
        self.X = [torch.tensor(data, dtype=torch.float32) for data in patients_data]
        self.y = torch.tensor(patients_outcomes, dtype=torch.float32)
        self.patient_ids = selected_ids
        
        # Calculate statistics for normalization
        all_values = np.vstack([data for data in patients_data])
        
        for i in range(all_values.shape[1]):
            column = all_values[:, i]
            valid_values = column[~np.isnan(column)]
            
            if len(valid_values) > 0:
                self.means.append(float(np.nanmean(valid_values)))
                self.stds.append(float(np.nanstd(valid_values) or 1.0))  # Avoid division by zero
                self.maxes.append(float(np.nanmax(valid_values)))
                self.mins.append(float(np.nanmin(valid_values)))
            else:
                self.means.append(0.0)
                self.stds.append(1.0)
                self.maxes.append(0.0)
                self.mins.append(0.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        """Get a single patient's data, processed into fixed-length time bins"""
        if self.temp_cache is not None and i in self.temp_cache:
            return self.temp_cache[i]
        
        # Get data for this patient
        ins = self.X[i]
        
        # Time is the first column
        time = ins[:, 0]
        
        # No static features in this implementation, but we'll keep the structure
        # for compatibility with DuETT
        x_static = torch.zeros(self.d_static_num())
        
        # Create time series tensor with dimensions [n_timesteps, features*2]
        # The second half of features is used as a binary mask for valid values
        x_ts = torch.zeros((self.n_timesteps, self.d_time_series_num()*2))
        
        # Last time point (max days)
        max_days = time[-1]
        
        # Process each time point
        for i_t, t in enumerate(time):
            # Determine which bin this time point belongs to
            bin_idx = self.n_timesteps - 1 if t == max_days else int(t / max_days * self.n_timesteps)
            
            # Process each lab value at this time point
            for i_lab in range(1, ins.shape[1]):
                x_i = ins[i_t, i_lab]
                if not torch.isnan(x_i).item():
                    # Normalize the value
                    norm_value = (x_i - self.means[i_lab]) / (self.stds[i_lab] + 1e-7)
                    x_ts[bin_idx, i_lab-1] = norm_value
                    # Mark this value as observed
                    x_ts[bin_idx, i_lab-1+self.d_time_series_num()] += 1
        
        # Calculate bin end times
        bin_ends = torch.arange(1, self.n_timesteps+1) / self.n_timesteps * max_days
        
        # Package the data
        x = (x_ts, x_static, bin_ends)
        y = self.y[i]
        
        # Cache if needed
        if self.temp_cache is not None:
            self.temp_cache[i] = (x, y)
        
        return x, y
    
    def d_static_num(self):
        """The total dimension of numeric static features"""
        return 0  # No static features in this implementation
    
    def d_time_series_num(self):
        """The total dimension of numeric time-series features"""
        return len(self.lab_columns)
    
    def d_target(self):
        return 1
    
    def pos_frac(self):
        if self.y is not None:
            return self.y.float().mean().item()
        return 0.5

# Collate function for batching
def collate_into_seqs(batch):
    xs, ys = zip(*batch)
    return zip(*xs), torch.tensor(ys)

class DialysisDataModule(pl.LightningDataModule):
    def __init__(self, data_path=None, use_temp_cache=False, batch_size=8, 
                 num_workers=1, prefetch_factor=2, target_col='outcome', 
                 seed=0, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.use_temp_cache = use_temp_cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_col = target_col
        self.seed = seed
        
        # Create datasets for each split
        self.ds_train = DialysisDataset('train', data_path=self.data_path, 
                                        use_temp_cache=use_temp_cache,
                                        target_col=target_col, seed=seed)
        self.ds_val = DialysisDataset('val', data_path=self.data_path, 
                                     use_temp_cache=use_temp_cache,
                                     target_col=target_col, seed=seed)
        self.ds_test = DialysisDataset('test', data_path=self.data_path, 
                                      use_temp_cache=use_temp_cache,
                                      target_col=target_col, seed=seed)

        self.prepare_data_per_node = False
        
        # DataLoader arguments
        self.dl_args = {'batch_size': self.batch_size, 'prefetch_factor': self.prefetch_factor,
                'collate_fn': collate_into_seqs, 'num_workers': num_workers}

    def setup(self, stage=None):
        if stage is None:
            self.ds_train.setup()
            self.ds_val.setup()
            self.ds_test.setup()
        elif stage == 'fit':
            self.ds_train.setup()
            self.ds_val.setup()
        elif stage == 'validate':
            self.ds_val.setup()
        elif stage == 'test':
            self.ds_test.setup()

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.ds_train, shuffle=True, **self.dl_args)

    def val_dataloader(self):
        return DataLoader(self.ds_val, **self.dl_args)

    def test_dataloader(self):
        return DataLoader(self.ds_test, **self.dl_args)
    
    def d_static_num(self):
        return self.ds_train.d_static_num()

    def d_time_series_num(self):
        return self.ds_train.d_time_series_num()

    def d_target(self):
        return self.ds_train.d_target()

    def pos_frac(self):
        return self.ds_train.pos_frac()