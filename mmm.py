import os
# Disable GPU/Metal usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_DISABLE_METAL'] = '1'

import sys
import streamlit as st
import inspect

# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Meridian MMM App",
    page_icon="📈",
    layout="wide"
)

# Now continue with other imports
import pandas as pd
import numpy as np
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import tensorflow_probability as tfp

# Import meridian
import meridian
from meridian.data.input_data import InputData as MeridianInputData
from meridian.model.prior_distribution import PriorDistribution
from meridian.model.spec import ModelSpec
from meridian.model.model import Meridian

# Debug info can come after set_page_config
st.sidebar.text(f"Meridian version: {meridian.__version__}")

def generate_sample_data(data_size, n_media_channels, n_extra_features):
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Generate dates
    start_date = pd.Timestamp('2023-01-01')
    dates = pd.date_range(start=start_date, periods=data_size, freq='W')
    
    # Generate media data
    media_data = np.random.uniform(0, 100, (data_size, n_media_channels))
    media_df = pd.DataFrame(
        media_data,
        columns=[f'channel_{i+1}' for i in range(n_media_channels)],
        index=dates
    )
    
    # Generate costs
    costs = np.random.uniform(5, 15, n_media_channels)
    costs_df = pd.DataFrame({
        'channel': [f'channel_{i+1}' for i in range(n_media_channels)],
        'cost_per_unit': costs
    })
    
    # Generate extra features
    extra_features = np.random.normal(0, 1, (data_size, n_extra_features))
    extra_features_df = pd.DataFrame(
        extra_features,
        columns=[f'feature_{i+1}' for i in range(n_extra_features)],
        index=dates
    )
    
    # Generate target variable
    base = 1000
    noise = np.random.normal(0, 50, data_size)
    target = base + np.sum(media_data * np.random.uniform(0.5, 2, n_media_channels), axis=1) + noise
    target_df = pd.DataFrame({'revenue': target}, index=dates)
    
    return media_df, extra_features_df, target_df, costs_df

def main():
    st.title("Media Mix Modeling with Google Meridian 📊")
    st.sidebar.title("Configuration")

    # Initialize variables that need to be shared across sections
    kpi_column = None
    kpi_type = None
    media_df = None
    target_df = None
    extra_features_df = None
    costs_df = None  # Added this since it's used in sample data

    # Sidebar options
    data_option = st.sidebar.selectbox(
        "Choose Data Source",
        ["Upload Data", "Use Sample Data"]
    )

    if data_option == "Use Sample Data":
        st.sidebar.subheader("Sample Data Parameters")
        data_size = st.sidebar.slider("Data Size", 50, 200, 100)
        n_media_channels = st.sidebar.slider("Number of Media Channels", 2, 10, 3)
        n_extra_features = st.sidebar.slider("Number of Extra Features", 1, 5, 2)
        
        # Generate sample data
        media_df, extra_features_df, target_df, costs_df = generate_sample_data(
            data_size=data_size,
            n_media_channels=n_media_channels,
            n_extra_features=n_extra_features
        )

        # Set default values for sample data
        kpi_column = 'revenue'  # Changed from 'target' to 'revenue'
        kpi_type = 'revenue'  # Changed from 'Revenue' to 'revenue'

        # Display sample data
        st.subheader("Sample Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Media Data")
            st.dataframe(media_df)
        
        with col2:
            st.write("Target Variable")
            st.line_chart(target_df)

    else:
        st.subheader("Upload Your Data")
        
        # File upload section
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                # Display the uploaded data
                st.write("Uploaded Data Preview:")
                st.dataframe(df.head())
                
                # Get column names for selection
                all_columns = df.columns.tolist()
                
                # Date column selection
                date_column = st.selectbox(
                    "Select Date Column",
                    all_columns,
                    help="Select the column containing dates"
                )
                
                # Convert date column to datetime and set as index
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                    df = df.set_index(date_column)
                except Exception as e:
                    st.error(f"Error processing date column: {str(e)}")
                    st.write("Please ensure your date column is in a valid format (e.g., YYYY-MM-DD)")
                    return
                
                # KPI selection
                st.subheader("KPI Configuration")
                kpi_column = st.selectbox(
                    "Select KPI Column",
                    all_columns,
                    help="Select the column containing your target metric"
                )
                
                # KPI type selection
                kpi_type = st.radio(
                    "KPI Type",
                    ["revenue", "non_revenue"],  # Changed from ["Revenue", "Non-Revenue"]
                    help="Select whether your KPI is revenue or another metric"
                )
                
                # If non-revenue, allow revenue per KPI specification
                revenue_per_kpi = None
                if kpi_type == "non_revenue":
                    revenue_per_kpi_option = st.radio(
                        "Revenue per KPI",
                        ["Constant Value", "Column in Data"],
                        help="Specify how to determine revenue per KPI"
                    )
                    
                    if revenue_per_kpi_option == "Constant Value":
                        revenue_per_kpi_value = st.number_input(
                            "Revenue per KPI Value",
                            min_value=0.0,
                            value=1.0,
                            help="Enter the constant value for revenue per KPI"
                        )
                        revenue_per_kpi = xr.DataArray(
                            np.ones((1, len(df))) * revenue_per_kpi_value,
                            dims=['geo', 'time'],
                            coords={
                                'geo': ['national'],
                                'time': df.index.strftime('%Y-%m-%d')  # Convert to string dates
                            },
                            name='revenue_per_kpi'
                        )
                    else:
                        revenue_per_kpi_column = st.selectbox(
                            "Select Revenue per KPI Column",
                            all_columns,
                            help="Select the column containing revenue per KPI values"
                        )
                        revenue_per_kpi = xr.DataArray(
                            df[revenue_per_kpi_column].values.reshape(1, -1),
                            dims=['geo', 'time'],
                            coords={
                                'geo': ['national'],
                                'time': df.index.strftime('%Y-%m-%d')  # Convert to string dates
                            },
                            name='revenue_per_kpi'
                        )
                
                # Media channels selection
                st.subheader("Media Channels")
                media_columns = st.multiselect(
                    "Select Media Channel Columns",
                    all_columns,
                    help="Select columns containing media spend data"
                )
                
                # Controls selection
                st.subheader("Control Variables")
                
                # Get available columns for controls (exclude date and KPI columns)
                available_control_columns = [col for col in all_columns if col not in [date_column, kpi_column]]
                
                if not available_control_columns:
                    st.warning("No columns available for control variables. Please ensure your data has columns other than the date and KPI columns.")
                    return
                
                control_columns = st.multiselect(
                    "Select Control Variable Columns",
                    available_control_columns,
                    help="Select columns containing control variables (excluding date and KPI columns)"
                )
                
                # Create data structures while preserving the index
                try:
                    # First validate the data
                    if not media_columns:
                        st.error("Please select at least one media channel column.")
                        return
                        
                    if not kpi_column:
                        st.error("Please select a KPI column.")
                        return
                        
                    # Create DataFrames with the date index
                    try:
                        target_df = pd.DataFrame({kpi_column: df[kpi_column]}, index=df.index)
                        media_df = pd.DataFrame(df[media_columns], index=df.index)
                        
                        # Handle control variables - create empty DataFrame if none selected
                        if control_columns:
                            extra_features_df = pd.DataFrame(df[control_columns], index=df.index)
                        else:
                            # Create empty DataFrame with same index if no controls selected
                            extra_features_df = pd.DataFrame(index=df.index)
                            st.info("No control variables selected. Proceeding without control variables.")
                    except KeyError as e:
                        st.error(f"Error accessing column(s): {str(e)}. Please ensure all selected columns exist in your data.")
                        return
                    
                    # Convert datetime index to string format for xarray
                    time_coords = df.index.strftime('%Y-%m-%d')
                    
                    # Create a common time coordinate as a pandas DatetimeIndex
                    time_coords = pd.DatetimeIndex(target_df.index)
                    
                    # Convert time coordinates to string format
                    time_coords_str = time_coords.strftime('%Y-%m-%d')
                    
                    # Debug time coordinate
                    st.text(f"\nTime coordinate type: {type(time_coords)}")
                    st.text(f"Time coordinate values: {time_coords}")
                    st.text(f"Time coordinate name: {time_coords.name}")

                    # KPI data: (n_geos, n_times)
                    kpi_xr = xr.DataArray(
                        target_df[kpi_column].values.reshape(1, -1),  # Reshape to (1, n_times)
                        dims=['geo', 'time'],
                        coords={
                            'geo': ['national'],
                            'time': ('time', time_coords_str)  # Use string format
                        },
                        name='kpi'
                    )

                    # Media data: (n_geos, n_times, n_media_channels)
                    media_xr = xr.DataArray(
                        media_df.values.reshape(1, -1, len(media_df.columns)),  # Reshape to (1, n_times, n_channels)
                        dims=['geo', 'media_time', 'media_channel'],
                        coords={
                            'geo': ['national'],
                            'media_time': ('media_time', time_coords_str),  # Use string format
                            'media_channel': media_df.columns
                        },
                        name='media'
                    )

                    # Controls data: (n_geos, n_times, n_controls)
                    if not control_columns:
                        # Create empty controls DataArray with proper dimensions
                        controls_xr = xr.DataArray(
                            np.zeros((1, len(time_coords), 0)),  # Shape: (1, n_times, 0)
                            dims=['geo', 'time', 'control_variable'],
                            coords={
                                'geo': ['national'],
                                'time': ('time', time_coords_str),  # Use string format
                                'control_variable': []
                            },
                            name='controls'
                        )
                    else:
                        controls_xr = xr.DataArray(
                            extra_features_df.values.reshape(1, -1, len(extra_features_df.columns)),  # Shape: (1, n_times, n_controls)
                            dims=['geo', 'time', 'control_variable'],
                            coords={
                                'geo': ['national'],
                                'time': ('time', time_coords_str),  # Use string format
                                'control_variable': extra_features_df.columns
                            },
                            name='controls'
                        )

                    # Media spend data: (n_media_channels,)
                    media_spend_xr = xr.DataArray(
                        np.ones(len(media_df.columns)),
                        dims=['media_channel'],
                        coords={'media_channel': media_df.columns},
                        name='media_spend'
                    )

                    # Population data: (n_geos,)
                    population_xr = xr.DataArray(
                        [1.0],
                        dims=['geo'],
                        coords={'geo': ['national']},
                        name='population'
                    )

                except Exception as e:
                    st.error(f"Error creating DataArrays: {str(e)}")
                    st.write("Error details:", str(e))
                    return
                
                # Display processed data
                st.subheader("Processed Data Overview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Media Data")
                    st.dataframe(media_df)
                
                with col2:
                    st.write("Target Variable")
                    st.line_chart(target_df)
                
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                st.write("Please ensure your file is in the correct format and try again.")
        else:
            st.info("Please upload a CSV file to proceed.")

    # Model configuration
    st.subheader("Model Configuration")
    col3, col4 = st.columns(2)
    
    with col3:
        roi_mu = st.number_input("ROI Prior Mean (mu)", 0.0, 2.0, 0.2)
        roi_sigma = st.number_input("ROI Prior Std Dev (sigma)", 0.1, 2.0, 0.9)
        n_chains = st.number_input("Number of Chains", 1, 7, 4)
    
    with col4:
        n_adapt = st.number_input("Number of Adaptation Steps", 100, 1000, 500)
        n_burnin = st.number_input("Number of Burn-in Steps", 100, 1000, 500)
        n_keep = st.number_input("Number of Samples to Keep", 500, 2000, 1000)

    if st.button("Train Model"):
        # First check if we have all required variables
        if kpi_column is None:
            st.error("KPI column is not defined. Please select a KPI column or use sample data.")
            return
        if kpi_type is None:
            st.error("KPI type is not defined. Please select a KPI type or use sample data.")
            return
        if media_df is None or target_df is None:
            st.error("Required data is missing. Please upload data or use sample data.")
            return

        with st.spinner("Training model..."):
            try:
                # First, ensure all DataFrames have datetime index and convert to proper string format
                media_df.index = pd.to_datetime(media_df.index).strftime('%Y-%m-%d')
                target_df.index = pd.to_datetime(target_df.index).strftime('%Y-%m-%d')
                if extra_features_df is not None:
                    extra_features_df.index = pd.to_datetime(extra_features_df.index).strftime('%Y-%m-%d')

                st.text("Creating InputData instance...")
                st.text(f"Media channels: {list(media_df.columns)}")
                st.text(f"Time range: {media_df.index[0]} to {media_df.index[-1]}")
                st.text(f"KPI column: {kpi_column}")
                st.text(f"KPI type: {kpi_type}")

                # Create xarray DataArrays with proper dimensions
                # First, ensure all indices are datetime
                target_df.index = pd.to_datetime(target_df.index)
                media_df.index = pd.to_datetime(media_df.index)
                if control_columns:
                    extra_features_df.index = pd.to_datetime(extra_features_df.index)

                # Create a common time coordinate as a pandas DatetimeIndex
                time_coords = pd.DatetimeIndex(target_df.index)
                
                # Convert time coordinates to string format
                time_coords_str = time_coords.strftime('%Y-%m-%d')
                
                # Debug time coordinate
                st.text(f"\nTime coordinate type: {type(time_coords)}")
                st.text(f"Time coordinate values: {time_coords}")
                st.text(f"Time coordinate name: {time_coords.name}")

                # KPI data: (n_geos, n_times)
                kpi_xr = xr.DataArray(
                    target_df[kpi_column].values.reshape(1, -1),  # Reshape to (1, n_times)
                    dims=['geo', 'time'],
                    coords={
                        'geo': ['national'],
                        'time': ('time', time_coords_str)  # Use string format
                    },
                    name='kpi'
                )

                # Media data: (n_geos, n_times, n_media_channels)
                media_xr = xr.DataArray(
                    media_df.values.reshape(1, -1, len(media_df.columns)),  # Reshape to (1, n_times, n_channels)
                    dims=['geo', 'media_time', 'media_channel'],
                    coords={
                        'geo': ['national'],
                        'media_time': ('media_time', time_coords_str),  # Use string format
                        'media_channel': media_df.columns
                    },
                    name='media'
                )

                # Controls data: (n_geos, n_times, n_controls)
                if not control_columns:
                    # Create empty controls DataArray with proper dimensions
                    controls_xr = xr.DataArray(
                        np.zeros((1, len(time_coords), 0)),  # Shape: (1, n_times, 0)
                        dims=['geo', 'time', 'control_variable'],
                        coords={
                            'geo': ['national'],
                            'time': ('time', time_coords_str),  # Use string format
                            'control_variable': []
                        },
                        name='controls'
                    )
                else:
                    controls_xr = xr.DataArray(
                        extra_features_df.values.reshape(1, -1, len(extra_features_df.columns)),  # Shape: (1, n_times, n_controls)
                        dims=['geo', 'time', 'control_variable'],
                        coords={
                            'geo': ['national'],
                            'time': ('time', time_coords_str),  # Use string format
                            'control_variable': extra_features_df.columns
                        },
                        name='controls'
                    )

                # Media spend data: (n_media_channels,)
                media_spend_xr = xr.DataArray(
                    np.ones(len(media_df.columns)),
                    dims=['media_channel'],
                    coords={'media_channel': media_df.columns},
                    name='media_spend'
                )

                # Population data: (n_geos,)
                population_xr = xr.DataArray(
                    [1.0],
                    dims=['geo'],
                    coords={'geo': ['national']},
                    name='population'
                )

                # Debug info
                st.text("\nData shapes and dimensions:")
                st.text(f"KPI: shape={kpi_xr.shape}, dims={kpi_xr.dims}")
                st.text(f"Media: shape={media_xr.shape}, dims={media_xr.dims}")
                st.text(f"Controls: shape={controls_xr.shape}, dims={controls_xr.dims}")
                st.text(f"Media spend: shape={media_spend_xr.shape}, dims={media_spend_xr.dims}")
                st.text(f"Population: shape={population_xr.shape}, dims={population_xr.dims}")

                # Verify time formats
                st.text("\nTime formats:")
                st.text(f"Time coordinate type: {type(time_coords)}")
                st.text(f"Time coordinate example: {time_coords[0]}")
                st.text(f"KPI time example: {kpi_xr.time.values[0]}")
                st.text(f"Media time example: {media_xr.media_time.values[0]}")
                st.text(f"Controls time example: {controls_xr.time.values[0]}")

                # Create InputData instance with proper structure
                try:
                    input_data = MeridianInputData(
                        kpi=kpi_xr,
                        kpi_type=kpi_type,
                        media=media_xr,
                        media_spend=media_spend_xr,
                        controls=controls_xr,
                        population=population_xr
                    )
                    st.text("Input data created successfully")
                except Exception as e:
                    st.error(f"Error creating InputData: {str(e)}")
                    raise
                
                # Model configuration
                st.text("Model configuration...")
                try:
                    prior = PriorDistribution(
                        roi_m=tfp.distributions.LogNormal(
                            loc=roi_mu,
                            scale=roi_sigma,
                            name='roi_multiplier'
                        )
                    )
                    model_spec = ModelSpec(prior=prior)
                    st.text("Model configuration successful")
                except Exception as e:
                    st.error(f"Error in model configuration: {str(e)}")
                    raise
                
                # Initialize model
                st.text("Initializing model...")
                try:
                    mmm = Meridian(
                        input_data=input_data,
                        model_spec=model_spec
                    )
                    st.text("Model initialization successful")
                except Exception as e:
                    st.error(f"Error in model initialization: {str(e)}")
                    raise
                
                # Debug info about methods
                st.text("Available methods:")
                st.text(f"sample_prior: {inspect.signature(mmm.sample_prior)}")
                st.text(f"sample_posterior: {inspect.signature(mmm.sample_posterior)}")

                # Sample from prior and posterior
                st.text("Sampling from prior...")
                try:
                    # Use n_draws instead of n_samples
                    mmm.sample_prior(n_draws=500)  # Changed from n_samples to n_draws
                    st.text("Prior sampling successful")
                except Exception as e:
                    st.error(f"Error in prior sampling: {str(e)}")
                    raise

                st.text("Sampling from posterior...")
                try:
                    mmm.sample_posterior(
                        n_chains=n_chains,
                        n_adapt=n_adapt,
                        n_burnin=n_burnin,
                        n_keep=n_keep,
                        seed=42
                    )
                    st.text("Posterior sampling successful")
                except Exception as e:
                    st.error(f"Error in posterior sampling: {str(e)}")
                    raise

                # Model diagnostics
                st.subheader("Model Results")
                try:
                    # First, import the necessary analyzer class
                    from meridian.analysis.analyzer import Analyzer
                    
                    # Create analyzer instance with error checking
                    try:
                        analyzer = Analyzer(mmm)
                        st.write("Successfully created analyzer instance")
                    except Exception as e:
                        st.error(f"Failed to create analyzer: {str(e)}")
                        st.text(f"MMM object type: {type(mmm)}")
                        st.text(f"MMM object attributes: {dir(mmm)}")
                        raise
                    
                    # Debug: Print analyzer methods
                    st.write("Available analyzer methods:")
                    st.write([method for method in dir(analyzer) if not method.startswith('_')])
                    
                    # Get expected vs actual data
                    st.write("Model Fit Analysis")
                    expected_vs_actual = analyzer.expected_vs_actual_data()
                    
                    # Get the data and reshape explicitly
                    dates = mmm.input_data.kpi.coords['time'].values
                    actual_values = expected_vs_actual['actual'].values.squeeze() # Shape should be (100,)
                    
                    # Debug: Check predicted values shape before aggregation
                    predicted_raw = expected_vs_actual['expected'].values.squeeze()
                    st.write(f"Raw predicted values shape: {predicted_raw.shape}")
                    
                    # Aggregate predicted values if multiple samples exist
                    if predicted_raw.ndim > 1:
                        # Assume the last dimension is samples, take the mean
                        predicted_values = predicted_raw.mean(axis=-1) 
                    else:
                        predicted_values = predicted_raw
                        
                    st.write(f"Final actual values shape: {actual_values.shape}")
                    st.write(f"Final predicted values shape: {predicted_values.shape}")

                    # 1. Time Series Plot
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=actual_values,
                        name='Actual',
                        mode='lines',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=predicted_values, # Use the aggregated predictions
                        name='Predicted (Mean)',
                        mode='lines',
                        line=dict(color='red')
                    ))
                    
                    if 'baseline' in expected_vs_actual:
                        baseline_values = expected_vs_actual['baseline'].values.squeeze()
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=baseline_values,
                            name='Baseline',
                            mode='lines',
                            line=dict(color='gray', dash='dash')
                        ))
                    
                    fig.update_layout(
                        title='Time Series: Actual vs Predicted Values',
                        xaxis_title='Date',
                        yaxis_title='KPI Value',
                        showlegend=True
                    )
                    st.plotly_chart(fig)
                    
                    # Calculate and display fit metrics using aggregated predictions
                    mse = np.mean((actual_values - predicted_values) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(actual_values - predicted_values))
                    # Handle potential division by zero or NaN in MAPE
                    mape_values = np.abs((actual_values - predicted_values) / actual_values)
                    mape = np.mean(mape_values[np.isfinite(mape_values)]) * 100 
                    r2 = 1 - np.sum((actual_values - predicted_values) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE (%)'],
                        'Value': [mse, rmse, mae, r2, mape]
                    })
                    st.write("Model Fit Metrics:")
                    st.table(metrics_df)
                    
                    # Get channel-specific metrics
                    try:
                        # Import tensorflow if not already imported
                        import tensorflow as tf 

                        # Get incremental outcome for each channel
                        incremental = analyzer.incremental_outcome(use_kpi=True)
                        
                        # Debug information
                        st.write("\nIncremental Outcome Structure:")
                        st.write(f"Type: {type(incremental)}")
                        # Use tf.shape for EagerTensor shape
                        if isinstance(incremental, tf.Tensor):
                            st.write(f"Shape: {tf.shape(incremental).numpy()}") 
                        else:
                            st.write(f"Shape: {incremental.shape}")

                        # Convert EagerTensor to NumPy array
                        if isinstance(incremental, tf.Tensor):
                            incremental_np = incremental.numpy()
                            st.write("Converted EagerTensor to NumPy array.")
                        else:
                            incremental_np = incremental # Assume it's already NumPy compatible

                        st.write(f"NumPy array shape: {incremental_np.shape}")

                        # Determine dimensions (assuming chains, draws/time, channels)
                        # Shape is (4, 1000, 3) - likely (chains, draws, channels)
                        if incremental_np.ndim == 3:
                            n_chains, n_draws, n_channels = incremental_np.shape
                            st.write(f"Interpreted dimensions: {n_chains} chains, {n_draws} draws, {n_channels} channels")
                            # Aggregate over chains and draws (axes 0 and 1)
                            total_channel_effects = np.sum(incremental_np, axis=(0, 1)) 
                            total_effect_all_channels = np.sum(total_channel_effects)
                        elif incremental_np.ndim == 2: # Maybe (time, channels)?
                             st.write(f"Interpreted dimensions: {incremental_np.shape[0]} time, {incremental_np.shape[1]} channels")
                             total_channel_effects = np.sum(incremental_np, axis=0)
                             total_effect_all_channels = np.sum(total_channel_effects)
                        else:
                            st.error(f"Unexpected number of dimensions in incremental outcome: {incremental_np.ndim}")
                            raise ValueError("Cannot interpret incremental outcome dimensions")

                        # Get channel names from the original media data
                        channel_names = mmm.input_data.media.coords['media_channel'].values
                        
                        # Ensure number of channels matches
                        if len(channel_names) != len(total_channel_effects):
                             st.error(f"Mismatch between channel names ({len(channel_names)}) and calculated effects ({len(total_channel_effects)})")
                             raise ValueError("Channel count mismatch")

                        # Create summary for each channel using aggregated effects
                        channel_summary = []
                        for i, channel in enumerate(channel_names):
                            channel_impact = total_channel_effects[i]
                            contribution_pct = (channel_impact / total_effect_all_channels * 100) if total_effect_all_channels != 0 else 0
                            channel_summary.append({
                                'Channel': channel,
                                'Total Impact': channel_impact,
                                'Contribution %': contribution_pct
                            })
                        
                        channel_df = pd.DataFrame(channel_summary)
                        
                        st.write("\nChannel Impact Summary:")
                        st.write(channel_df)
                        
                        # Create contribution pie chart
                        pie_fig = go.Figure(data=[go.Pie(
                            labels=channel_df['Channel'],
                            values=channel_df['Contribution %'],
                            hole=.3
                        )])
                        pie_fig.update_layout(title='Channel Contribution Distribution')
                        st.plotly_chart(pie_fig)
                        
                    except Exception as e:
                        st.warning(f"Could not analyze channel impacts: {str(e)}")
                        st.write("Error details:", str(e))
                        import traceback
                        st.text(f"Detailed error:\n{traceback.format_exc()}")
                        
                    # Get adstock parameters
                    st.write("\nAdstock Parameters:")
                    try:
                        adstock = analyzer.adstock_decay()
                        # Debug: Show type and data
                        st.write(f"Adstock data type: {type(adstock)}")
                        st.write(adstock)
                    except Exception as e:
                        st.warning(f"Could not retrieve adstock parameters: {str(e)}")
                        st.write("Error details:", str(e))
                        import traceback
                        st.text(f"Detailed error:\n{traceback.format_exc()}")
                        
                    # Get baseline metrics
                    st.write("\nBaseline Metrics:")
                    try:
                        baseline = analyzer.baseline_summary_metrics()
                        # Debug: Show type and data
                        st.write(f"Baseline data type: {type(baseline)}")
                        st.write(baseline)
                        
                        # Extract baseline contribution if possible
                        if isinstance(baseline, (pd.DataFrame, pd.Series)):
                            if 'contribution_percentage' in baseline:
                                 st.write(f"Baseline Contribution: {baseline['contribution_percentage'].iloc[0]:.2f}%")
                            elif 'contribution' in baseline:
                                 # Attempt to calculate percentage if total effect is available
                                 if 'total_effect_all_channels' in locals(): # Check if calculated earlier
                                    baseline_contrib = baseline['contribution'].iloc[0]
                                    baseline_pct = (baseline_contrib / (baseline_contrib + total_effect_all_channels)) * 100 if (baseline_contrib + total_effect_all_channels) !=0 else 0
                                    st.write(f"Baseline Contribution: {baseline_pct:.2f}%")
                                 else:
                                     st.write(f"Baseline Absolute Contribution: {baseline['contribution'].iloc[0]}")

                    except Exception as e:
                        st.warning(f"Could not retrieve baseline metrics: {str(e)}")
                        st.write("Error details:", str(e))
                        import traceback
                        st.text(f"Detailed error:\n{traceback.format_exc()}")
                        
                    # --- Adstock Visualization ---
                    st.subheader("Media Adstock Effects")
                    try:
                        adstock_params = analyzer.adstock_decay()
                        st.write("Adstock Parameters:")
                        st.write(adstock_params) # Show the parameters first

                        # Visualize decay for each channel
                        impulse = np.zeros(52) # Simulate an impulse spend over 52 weeks
                        impulse[0] = 100 
                        time_lags = np.arange(len(impulse))

                        adstock_fig = go.Figure()

                        # Get unique channels
                        channels = adstock_params['channel'].unique()
                        
                        for channel in channels:
                            # Get the mean decay values for this channel
                            channel_data = adstock_params[adstock_params['channel'] == channel]
                            # Use the mean values as decay rates
                            decay_rates = channel_data['mean'].values
                            
                            # Calculate adstocked impulse for each decay rate
                            for i, decay_rate in enumerate(decay_rates):
                                adstocked_impulse = np.zeros_like(impulse)
                                for t in range(len(impulse)):
                                    if t == 0:
                                        adstocked_impulse[t] = impulse[t]
                                    else:
                                        # Geometric decay formula
                                        adstocked_impulse[t] = impulse[t] + decay_rate * adstocked_impulse[t-1] 
                                
                                time_unit = channel_data.iloc[i]['time_units']
                                distribution = channel_data.iloc[i]['distribution']
                                adstock_fig.add_trace(go.Scatter(
                                    x=time_lags,
                                    y=adstocked_impulse,
                                    mode='lines',
                                    name=f'{channel} (t={time_unit:.1f}, {distribution})'
                                ))

                        adstock_fig.update_layout(
                            title='Adstock Effect Over Time (Impulse Response)',
                            xaxis_title='Weeks After Impulse Spend',
                            yaxis_title='Adstocked Effect',
                            showlegend=True
                        )
                        st.plotly_chart(adstock_fig)
                    except Exception as e:
                        st.warning(f"Could not retrieve or plot adstock parameters: {str(e)}")
                        st.write("Error details:", str(e))
                        # import traceback # Uncomment for detailed trace
                        # st.text(f"Detailed error:\n{traceback.format_exc()}")

                    # --- Saturation Visualization (Response & Hill Curves) ---
                    st.subheader("Media Saturation Effects")
                    try:
                        response_curves_data = analyzer.response_curves(use_kpi=True)
                        st.write("Response Curves Data:")
                        # Plot response curves for each channel
                        for channel_name, curve_data in response_curves_data.items():
                            st.write(f"\nProcessing Response Curve for {channel_name}:")
                            st.write(f"Type: {type(curve_data)}")
                            st.write(f"Shape: {curve_data.shape if hasattr(curve_data, 'shape') else 'N/A'}")
                            
                            try:
                                if hasattr(curve_data, 'values'): # Handle xarray DataArray
                                    curve_fig = go.Figure()
                                    
                                    # Get the coordinate values for x-axis (spend or input)
                                    if 'spend' in curve_data.coords:
                                        x_values = curve_data.coords['spend'].values
                                        x_label = 'Spend'
                                    elif 'input' in curve_data.coords:
                                        x_values = curve_data.coords['input'].values
                                        x_label = 'Input'
                                    else: # Fallback
                                        x_values = np.arange(len(curve_data.values))
                                        x_label = 'Input Level'
                                    
                                    y_values = curve_data.values
                                    
                                    curve_fig.add_trace(go.Scatter(
                                        x=x_values,
                                        y=y_values,
                                        name=f'{channel_name} (Modeled Response)',
                                        mode='lines'
                                    ))
                                    
                                    curve_fig.update_layout(
                                        title=f'Saturation Curve for {channel_name}',
                                        xaxis_title=x_label,
                                        yaxis_title='KPI Response',
                                        showlegend=True
                                    )
                                    st.plotly_chart(curve_fig)
                                else:
                                     st.warning(f"Could not plot response curve for {channel_name} - unexpected data structure.")

                            except Exception as plot_e:
                                st.warning(f"Could not plot response curve for {channel_name}: {str(plot_e)}")
                                st.write("Data structure:", curve_data)

                        # Additionally, try to plot Hill curves if the method exists
                        if hasattr(analyzer, 'hill_curves'):
                             st.write("\nHill Curve Parameters (Saturation Model):")
                             try:
                                hill_params = analyzer.hill_curves()
                                st.write(hill_params)

                                # Create a figure for the response curves
                                hill_fig = go.Figure()
                                
                                # Get unique channels
                                channels = hill_params['channel'].unique()
                                
                                for channel in channels:
                                    # Get data for this channel
                                    channel_data = hill_params[hill_params['channel'] == channel]
                                    
                                    # Separate prior and posterior data
                                    prior_data = channel_data[channel_data['distribution'] == 'prior']
                                    posterior_data = channel_data[channel_data['distribution'] == 'posterior']
                                    
                                    # Plot prior curve
                                    hill_fig.add_trace(go.Scatter(
                                        x=prior_data['media_units'],
                                        y=prior_data['mean'],
                                        name=f'{channel} (Prior)',
                                        mode='lines',
                                        line=dict(color='red', dash='dash')
                                    ))
                                    
                                    # Plot posterior curve
                                    hill_fig.add_trace(go.Scatter(
                                        x=posterior_data['media_units'],
                                        y=posterior_data['mean'],
                                        name=f'{channel} (Posterior)',
                                        mode='lines',
                                        line=dict(color='blue')
                                    ))
                                    
                                    # Add confidence intervals if available
                                    if 'ci_lo' in posterior_data.columns and 'ci_hi' in posterior_data.columns:
                                        hill_fig.add_trace(go.Scatter(
                                            x=posterior_data['media_units'],
                                            y=posterior_data['ci_hi'],
                                            fill=None,
                                            mode='lines',
                                            line=dict(color='blue', width=0),
                                            showlegend=False
                                        ))
                                        hill_fig.add_trace(go.Scatter(
                                            x=posterior_data['media_units'],
                                            y=posterior_data['ci_lo'],
                                            fill='tonexty',
                                            mode='lines',
                                            line=dict(color='blue', width=0),
                                            fillcolor='rgba(0,0,255,0.1)',
                                            showlegend=False
                                        ))
                                    
                                    hill_fig.update_layout(
                                        title='Media Response Curves',
                                        xaxis_title='Media Units',
                                        yaxis_title='Response',
                                        showlegend=True
                                    )
                                    st.plotly_chart(hill_fig)

                             except Exception as hill_e:
                                  st.warning(f"Could not retrieve or plot Hill curves: {str(hill_e)}")
                                  st.write("Error details:", str(hill_e))

                    except Exception as e:
                        st.warning(f"Could not retrieve or plot saturation curves: {str(e)}")
                        st.write("Error details:", str(e))
                        # import traceback # Uncomment for detailed trace
                        # st.text(f"Detailed error:\n{traceback.format_exc()}")

                except Exception as e:
                    st.error(f"Error in model analysis: {str(e)}")
                    import traceback
                    st.text(f"Detailed error:\n{traceback.format_exc()}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
                
                # Let's try to get more information about the class
                st.text("\nInputData class information:")
                from meridian.data.input_data import InputData
                st.text(f"InputData class docstring: {InputData.__doc__}")
                st.text(f"InputData class annotations: {getattr(InputData, '__annotations__', 'No annotations')}")

if __name__ == "__main__":
    main() 
