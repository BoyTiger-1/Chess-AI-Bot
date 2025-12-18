"""
Customer Intelligence Engine
Provides advanced customer analytics including segmentation, RFM analysis, churn prediction,
customer journey mapping, lifetime value calculation, and behavioral clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SegmentationMethod(Enum):
    """Customer segmentation methods."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    RFM_BASED = "rfm_based"
    BEHAVIORAL = "behavioral"
    PSYCHOGRAPHIC = "psychographic"
    VALUE_BASED = "value_based"


class ChurnModel(Enum):
    """Churn prediction model types."""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"


@dataclass
class CustomerSegment:
    """Customer segment definition."""
    segment_id: int
    segment_name: str
    size: int
    percentage: float
    characteristics: Dict[str, Any]
    key_metrics: Dict[str, float]
    recommended_actions: List[str]
    priority: str  # "high", "medium", "low"


@dataclass
class ChurnPredictionResult:
    """Churn prediction result."""
    model_type: ChurnModel
    churn_probability: np.ndarray
    churn_prediction: np.ndarray
    feature_importance: Dict[str, float]
    model_performance: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    customer_segments: Dict[str, Any]


@dataclass
class CLVResult:
    """Customer Lifetime Value result."""
    clv_scores: pd.DataFrame
    average_clv: float
    clv_distribution: Dict[str, float]
    predictive_accuracy: Dict[str, float]
    segment_analysis: Dict[str, Any]
    actionable_insights: List[str]


class CustomerIntelligenceEngine:
    """
    Advanced Customer Intelligence Engine
    Provides comprehensive customer analytics including segmentation, churn prediction,
    lifetime value calculation, and behavioral analysis.
    """
    
    def __init__(self, current_date: Optional[datetime] = None):
        """Initialize Customer Intelligence Engine."""
        self.current_date = current_date or datetime.now()
        self.rfm_data = None
        self.segment_models = {}
        self.churn_models = {}
        
    def calculate_rfm_metrics(self, 
                            transactions: pd.DataFrame,
                            customer_id_col: str = 'customer_id',
                            date_col: str = 'transaction_date',
                            amount_col: str = 'amount',
                            current_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics.
        
        Args:
            transactions: DataFrame with transaction data
            customer_id_col: Column name for customer IDs
            date_col: Column name for transaction dates
            amount_col: Column name for transaction amounts
            current_date: Reference date for recency calculation
        """
        if current_date is None:
            current_date = self.current_date
        
        # Ensure date column is datetime
        transactions[date_col] = pd.to_datetime(transactions[date_col])
        
        # Calculate RFM metrics
        rfm = transactions.groupby(customer_id_col).agg({
            date_col: lambda x: (current_date - x.max()).days,  # Recency
            customer_id_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']
        
        # Add additional metrics
        last_transaction = transactions.groupby(customer_id_col)[date_col].max().reset_index()
        last_transaction.columns = [customer_id_col, 'last_transaction_date']
        
        avg_days_between = transactions.groupby(customer_id_col).apply(
            lambda x: (x[date_col].max() - x[date_col].min()).days / (x[customer_id_col].count() - 1) 
            if x[customer_id_col].count() > 1 else 0
        ).reset_index()
        avg_days_between.columns = [customer_id_col, 'avg_days_between']
        
        # Merge additional metrics
        rfm = rfm.merge(last_transaction, on=customer_id_col)
        rfm = rfm.merge(avg_days_between, on=customer_id_col)
        
        # Calculate derived metrics
        rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
        rfm['days_since_first_purchase'] = (current_date - transactions.groupby(customer_id_col)[date_col].min()).dt.days
        
        return rfm
    
    def create_rfm_segments(self, 
                          rfm_data: pd.DataFrame,
                          method: str = "quartile") -> pd.DataFrame:
        """
        Create customer segments based on RFM analysis.
        
        Args:
            rfm_data: DataFrame with RFM metrics
            method: Segmentation method ("quartile", "decile", "custom")
        """
        rfm_segmented = rfm_data.copy()
        
        if method == "quartile":
            # Create quartiles for R, F, M
            rfm_segmented['R_score'] = pd.qcut(rfm_data['recency'], 4, labels=[4, 3, 2, 1])
            rfm_segmented['F_score'] = pd.qcut(rfm_data['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
            rfm_segmented['M_score'] = pd.qcut(rfm_data['monetary'], 4, labels=[1, 2, 3, 4])
        
        elif method == "decile":
            # Create deciles for R, F, M
            rfm_segmented['R_score'] = pd.qcut(rfm_data['recency'], 10, labels=list(range(10, 0, -1)))
            rfm_segmented['F_score'] = pd.qcut(rfm_data['frequency'].rank(method='first'), 10, labels=list(range(1, 11)))
            rfm_segmented['M_score'] = pd.qcut(rfm_data['monetary'], 10, labels=list(range(1, 11)))
        
        # Calculate RFM Score
        rfm_segmented['RFM_score'] = (
            rfm_segmented['R_score'].astype(str) + 
            rfm_segmented['F_score'].astype(str) + 
            rfm_segmented['M_score'].astype(str)
        )
        
        # Define customer segments based on RFM score
        segment_mapping = {
            '444': 'Champions',
            '443': 'Champions',
            '434': 'Loyal Customers',
            '343': 'Potential Loyalists',
            '344': 'Potential Loyalists',
            '433': 'Recent Customers',
            '443': 'Promising',
            '334': 'Need Attention',
            '343': 'Need Attention',
            '244': 'About To Sleep',
            '243': 'At Risk',
            '233': 'Cannot Lose Them',
            '144': 'Hibernating',
            '143': 'Lost'
        }
        
        # Create segment column
        rfm_segmented['segment'] = rfm_segmented['RFM_score'].map(
            lambda x: segment_mapping.get(x, 'Others')
        )
        
        # Calculate segment statistics
        segment_stats = rfm_segmented.groupby('segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'avg_order_value': 'mean'
        }).round(2)
        
        segment_stats.columns = ['count', 'avg_recency', 'avg_frequency', 'avg_monetary', 'avg_aov']
        segment_stats['percentage'] = (segment_stats['count'] / len(rfm_segmented) * 100).round(2)
        
        return rfm_segmented, segment_stats
    
    def advanced_customer_segmentation(self,
                                     customer_data: pd.DataFrame,
                                     features: List[str],
                                     method: SegmentationMethod = SegmentationMethod.KMEANS,
                                     n_clusters: int = 5,
                                     scaling_method: str = "standard") -> Dict[str, Any]:
        """
        Advanced customer segmentation using multiple clustering algorithms.
        
        Args:
            customer_data: DataFrame with customer data
            features: List of feature columns to use for clustering
            method: Clustering algorithm
            n_clusters: Number of clusters
            scaling_method: Feature scaling method
        """
        # Prepare features
        X = customer_data[features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        else:
            scaler = None
        
        if scaler:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Apply clustering algorithm
        if method == SegmentationMethod.KMEANS:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(X_scaled)
            cluster_centers = clusterer.cluster_centers_
            
        elif method == SegmentationMethod.HIERARCHICAL:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X_scaled)
            cluster_centers = None
            
        elif method == SegmentationMethod.DBSCAN:
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # Calculate cluster centers for DBSCAN
            cluster_centers = []
            for i in range(n_clusters):
                cluster_points = X_scaled[cluster_labels == i]
                if len(cluster_points) > 0:
                    center = np.mean(cluster_points, axis=0)
                    cluster_centers.append(center)
            cluster_centers = np.array(cluster_centers) if cluster_centers else None
            
        elif method == SegmentationMethod.GAUSSIAN_MIXTURE:
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(X_scaled)
            cluster_centers = clusterer.means_
            
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Calculate clustering metrics
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
        else:
            silhouette_avg = 0
            calinski_harabasz = 0
        
        # Create customer segments
        customer_data_clustered = customer_data.copy()
        customer_data_clustered['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(
            customer_data_clustered, cluster_labels, features, X_scaled
        )
        
        # Feature importance analysis
        feature_importance = self._calculate_feature_importance(
            X_scaled, cluster_labels, features
        )
        
        return {
            'customer_data': customer_data_clustered,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'cluster_analysis': cluster_analysis,
            'feature_importance': feature_importance,
            'clustering_metrics': {
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_harabasz,
                'n_clusters': n_clusters,
                'method': method.value
            },
            'scaler': scaler
        }
    
    def predict_customer_churn(self,
                             customer_data: pd.DataFrame,
                             churn_column: str = 'churned',
                             features: Optional[List[str]] = None,
                             model_type: ChurnModel = ChurnModel.RANDOM_FOREST,
                             test_size: float = 0.2) -> ChurnPredictionResult:
        """
        Predict customer churn using machine learning models.
        
        Args:
            customer_data: DataFrame with customer data
            churn_column: Name of churn indicator column
            features: List of feature columns
            model_type: Churn prediction model type
            test_size: Test split size
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        # Prepare features
        if features is None:
            features = [col for col in customer_data.columns if col != churn_column]
        
        X = customer_data[features].fillna(0)
        y = customer_data[churn_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        if model_type == ChurnModel.RANDOM_FOREST:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == ChurnModel.LOGISTIC_REGRESSION:
            model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Make predictions
        churn_prob = model.predict_proba(X_test)[:, 1]
        churn_pred = model.predict(X_test)
        
        # Calculate performance metrics
        auc_score = roc_auc_score(y_test, churn_prob)
        class_report = classification_report(y_test, churn_pred, output_dict=True)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(features, model.feature_importances_))
        else:
            feature_importance = dict(zip(features, np.abs(model.coef_[0])))
        
        # Generate insights
        churn_rate = y.mean()
        insights = [
            f"Overall churn rate: {churn_rate:.2%}",
            f"Model AUC score: {auc_score:.3f}",
            f"Top predictor: {max(feature_importance, key=feature_importance.get)}"
        ]
        
        # Generate recommendations
        recommendations = [
            "Implement targeted retention campaigns for high-risk customers",
            "Develop personalized engagement strategies",
            "Create early warning systems for customer satisfaction",
            "Focus on improving product/service quality",
            "Enhance customer support and service"
        ]
        
        return ChurnPredictionResult(
            model_type=model_type,
            churn_probability=churn_prob,
            churn_prediction=churn_pred,
            feature_importance=feature_importance,
            model_performance={
                'auc_score': auc_score,
                'accuracy': class_report['accuracy'],
                'precision': class_report['1']['precision'],
                'recall': class_report['1']['recall'],
                'f1_score': class_report['1']['f1-score']
            },
            insights=insights,
            recommendations=recommendations,
            customer_segments=self._segment_by_churn_risk(customer_data, churn_prob)
        )
    
    def calculate_customer_lifetime_value(self,
                                        transactions: pd.DataFrame,
                                        customer_data: pd.DataFrame,
                                        time_horizon: int = 24,  # months
                                        discount_rate: float = 0.1,
                                        retention_rate: float = 0.8,
                                        cost_per_customer: float = 50) -> CLVResult:
        """
        Calculate Customer Lifetime Value using multiple methods.
        
        Args:
            transactions: Transaction history
            customer_data: Customer attributes
            time_horizon: CLV calculation horizon in months
            discount_rate: Discount rate for future cash flows
            retention_rate: Monthly retention rate
            cost_per_customer: Cost to serve per customer per period
        """
        # Calculate historical CLV metrics
        customer_metrics = transactions.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        customer_metrics.columns = ['customer_id', 'total_spend', 'avg_order_value', 'order_frequency', 'first_purchase', 'last_purchase']
        
        # Calculate customer age in months
        current_date = self.current_date
        customer_metrics['customer_age_months'] = (
            current_date - customer_metrics['first_purchase']
        ).dt.days / 30.44
        
        customer_metrics['purchase_frequency_monthly'] = (
            customer_metrics['order_frequency'] / customer_metrics['customer_age_months']
        ).fillna(0)
        
        # Method 1: Historical CLV
        customer_metrics['historical_clv'] = customer_metrics['total_spend']
        
        # Method 2: Predictive CLV using retention and frequency
        monthly_discount_rate = (1 + discount_rate) ** (1/12) - 1
        retention_factor = retention_rate ** np.arange(1, time_horizon + 1)
        discount_factors = 1 / (1 + monthly_discount_rate) ** np.arange(1, time_horizon + 1)
        present_value_factors = retention_factor * discount_factors
        
        predicted_clv = (
            customer_metrics['purchase_frequency_monthly'] * 
            customer_metrics['avg_order_value'] * 
            np.sum(present_value_factors) - 
            cost_per_customer * np.sum(discount_factors)
        )
        
        customer_metrics['predicted_clv'] = predicted_clv.clip(lower=0)
        
        # Method 3: Cohort-based CLV
        cohort_clv = self._calculate_cohort_clv(transactions, time_horizon, monthly_discount_rate)
        customer_metrics = customer_metrics.merge(cohort_clv, on='customer_id', how='left')
        
        # Segment customers by CLV
        clv_percentiles = np.percentile(customer_metrics['predicted_clv'].dropna(), [20, 40, 60, 80])
        customer_metrics['clv_segment'] = pd.cut(
            customer_metrics['predicted_clv'],
            bins=[-np.inf] + list(clv_percentiles) + [np.inf],
            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        )
        
        # Calculate distribution statistics
        clv_distribution = {
            'mean': customer_metrics['predicted_clv'].mean(),
            'median': customer_metrics['predicted_clv'].median(),
            'std': customer_metrics['predicted_clv'].std(),
            'percentile_25': customer_metrics['predicted_clv'].quantile(0.25),
            'percentile_75': customer_metrics['predicted_clv'].quantile(0.75),
            'total_customers': len(customer_metrics),
            'profitable_customers': len(customer_metrics[customer_metrics['predicted_clv'] > cost_per_customer])
        }
        
        # Segment analysis
        segment_analysis = customer_metrics.groupby('clv_segment').agg({
            'customer_id': 'count',
            'predicted_clv': ['mean', 'sum'],
            'purchase_frequency_monthly': 'mean',
            'avg_order_value': 'mean'
        }).round(2)
        
        segment_analysis.columns = ['count', 'avg_clv', 'total_clv', 'avg_frequency', 'avg_aov']
        
        return CLVResult(
            clv_scores=customer_metrics,
            average_clv=clv_distribution['mean'],
            clv_distribution=clv_distribution,
            predictive_accuracy={
                'model_correlation': customer_metrics[['historical_clv', 'predicted_clv']].corr().iloc[0, 1],
                'prediction_range': (customer_metrics['predicted_clv'].min(), customer_metrics['predicted_clv'].max())
            },
            segment_analysis=segment_analysis.to_dict(),
            actionable_insights=[
                f"Focus on {clv_distribution['profitable_customers']} high-value customers",
                f"Top CLV segment contributes {segment_analysis.loc['High', 'total_clv']:.0f} in total value",
                "Implement retention strategies for medium-value customers",
                "Consider acquisition strategies for low-CLV segments"
            ]
        )
    
    def customer_journey_analysis(self,
                                events: pd.DataFrame,
                                customer_id_col: str = 'customer_id',
                                event_col: str = 'event_type',
                                timestamp_col: str = 'timestamp',
                                journey_stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze customer journey patterns and touchpoints.
        
        Args:
            events: DataFrame with customer interaction events
            customer_id_col: Customer ID column
            event_col: Event type column
            timestamp_col: Timestamp column
            journey_stages: Defined journey stages
        """
        # Prepare data
        events[timestamp_col] = pd.to_datetime(events[timestamp_col])
        events = events.sort_values([customer_id_col, timestamp_col])
        
        # Define default journey stages if not provided
        if journey_stages is None:
            journey_stages = ['awareness', 'consideration', 'purchase', 'retention', 'advocacy']
        
        # Create customer journey paths
        journey_paths = events.groupby(customer_id_col)[event_col].apply(
            lambda x: ' > '.join(x.unique())
        ).reset_index()
        journey_paths.columns = [customer_id_col, 'journey_path']
        
        # Analyze conversion funnels
        funnel_analysis = self._analyze_conversion_funnel(events, customer_id_col, event_col)
        
        # Calculate journey metrics
        journey_metrics = self._calculate_journey_metrics(events, customer_id_col, timestamp_col)
        
        # Identify common paths
        path_frequency = journey_paths['journey_path'].value_counts()
        
        # Time-based analysis
        temporal_analysis = self._analyze_journey_timing(events, customer_id_col, timestamp_col)
        
        # Drop-off analysis
        dropoff_analysis = self._analyze_dropoffs(events, customer_id_col, event_col, journey_stages)
        
        return {
            'journey_paths': journey_paths,
            'funnel_analysis': funnel_analysis,
            'journey_metrics': journey_metrics,
            'common_paths': path_frequency.head(20).to_dict(),
            'temporal_analysis': temporal_analysis,
            'dropoff_analysis': dropoff_analysis,
            'recommendations': self._generate_journey_recommendations(funnel_analysis, dropoff_analysis)
        }
    
    def behavioral_clustering(self,
                            customer_data: pd.DataFrame,
                            behavioral_features: List[str],
                            n_clusters: int = 5) -> Dict[str, Any]:
        """
        Perform behavioral clustering to identify customer behavior patterns.
        
        Args:
            customer_data: Customer data
            behavioral_features: Features related to customer behavior
            n_clusters: Number of behavioral clusters
        """
        # Prepare behavioral data
        behavior_data = customer_data[behavioral_features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        behavior_scaled = scaler.fit_transform(behavior_data)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(behavior_scaled)
        
        # Analyze clusters
        customer_data['behavioral_cluster'] = clusters
        cluster_analysis = customer_data.groupby('behavioral_cluster')[behavioral_features].mean()
        
        # Calculate cluster characteristics
        cluster_sizes = customer_data['behavioral_cluster'].value_counts().sort_index()
        cluster_percentages = (cluster_sizes / len(customer_data) * 100).round(2)
        
        # Define cluster personas
        personas = self._define_cluster_personas(cluster_analysis, behavioral_features)
        
        # Calculate within-cluster similarity
        silhouette_avg = silhouette_score(behavior_scaled, clusters)
        
        return {
            'customer_data': customer_data,
            'behavioral_clusters': clusters,
            'cluster_analysis': cluster_analysis,
            'cluster_sizes': cluster_sizes.to_dict(),
            'cluster_percentages': cluster_percentages.to_dict(),
            'personas': personas,
            'silhouette_score': silhouette_avg,
            'feature_importance': self._calculate_behavioral_importance(behavior_scaled, clusters, behavioral_features)
        }
    
    def market_basket_analysis(self,
                             transactions: pd.DataFrame,
                             customer_id_col: str = 'customer_id',
                             product_col: str = 'product_name',
                             transaction_id_col: str = 'transaction_id') -> Dict[str, Any]:
        """
        Perform market basket analysis to identify product associations.
        
        Args:
            transactions: Transaction data with products
            customer_id_col: Customer ID column
            product_col: Product column
            transaction_id_col: Transaction ID column
        """
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder
            
            # Create transaction matrix
            transactions_list = transactions.groupby(transaction_id_col)[product_col].apply(
                lambda x: x.unique().tolist()
            ).tolist()
            
            # Transform to binary matrix
            te = TransactionEncoder()
            te_ary = te.fit(transactions_list).transform(transactions_list)
            basket_df = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Find frequent itemsets
            frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)
            
            # Generate association rules
            if len(frequent_itemsets) > 0:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
                
                # Sort by lift
                rules = rules.sort_values('lift', ascending=False)
                
                # Calculate metrics
                support_values = frequent_itemsets['support'].describe()
                confidence_values = rules['confidence'].describe() if len(rules) > 0 else pd.Series()
                
                return {
                    'frequent_itemsets': frequent_itemsets.to_dict('records'),
                    'association_rules': rules.head(20).to_dict('records') if len(rules) > 0 else [],
                    'basket_size_distribution': basket_df.sum(axis=1).describe().to_dict(),
                    'item_popularity': basket_df.sum().sort_values(ascending=False).to_dict(),
                    'analysis_summary': {
                        'total_items': len(basket_df.columns),
                        'total_transactions': len(basket_df),
                        'avg_basket_size': basket_df.sum(axis=1).mean(),
                        'unique_products': len(basket_df.columns),
                        'frequent_itemsets_count': len(frequent_itemsets),
                        'association_rules_count': len(rules)
                    }
                }
            else:
                return {
                    'error': 'No frequent itemsets found with minimum support threshold',
                    'recommendations': ['Lower the minimum support threshold', 'Check data quality']
                }
                
        except ImportError:
            # Fallback analysis without mlxtend
            return self._simple_market_basket_analysis(transactions, customer_id_col, product_col, transaction_id_col)
    
    # Helper methods
    
    def _analyze_clusters(self, data: pd.DataFrame, labels: np.ndarray, features: List[str], scaled_data: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        analysis = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            analysis[f'cluster_{cluster_id}'] = {
                'size': np.sum(cluster_mask),
                'percentage': np.mean(cluster_mask) * 100,
                'characteristics': {
                    feature: {
                        'mean': cluster_data[feature].mean(),
                        'std': cluster_data[feature].std(),
                        'median': cluster_data[feature].median()
                    } for feature in features
                }
            }
        
        return analysis
    
    def _calculate_feature_importance(self, scaled_data: np.ndarray, labels: np.ndarray, features: List[str]) -> Dict[str, float]:
        """Calculate feature importance for clustering."""
        importance = {}
        
        for i, feature in enumerate(features):
            # Calculate variance between clusters
            cluster_means = []
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_mean = scaled_data[cluster_mask, i].mean()
                cluster_means.append(cluster_mean)
            
            # Variance of cluster means as importance measure
            importance[feature] = np.var(cluster_means)
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    def _segment_by_churn_risk(self, customer_data: pd.DataFrame, churn_probability: np.ndarray) -> Dict[str, str]:
        """Segment customers by churn risk."""
        risk_scores = pd.DataFrame({
            'customer_id': customer_data.index,
            'churn_probability': churn_probability
        })
        
        # Define risk segments
        risk_scores['risk_segment'] = pd.cut(
            risk_scores['churn_probability'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        )
        
        return risk_scores.set_index('customer_id')['risk_segment'].to_dict()
    
    def _calculate_cohort_clv(self, transactions: pd.DataFrame, time_horizon: int, discount_rate: float) -> pd.DataFrame:
        """Calculate cohort-based CLV."""
        # Simplified cohort analysis
        customers_cohort = transactions.groupby('customer_id')['transaction_date'].min().dt.to_period('M')
        customers_cohort.name = 'cohort_group'
        
        # Calculate monthly customer activity
        customers_monthly = transactions.set_index('transaction_date').groupby(['customer_id', pd.Grouper(freq='M')]).size().unstack(fill_value=0)
        
        # Merge cohort information
        customers_cohort_data = customers_monthly.merge(
            customers_cohort, 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        
        # Calculate retention and CLV
        cohort_table = customers_cohort_data.groupby('cohort_group').size()
        
        return pd.DataFrame({'customer_id': cohort_table.index, 'cohort_clv': cohort_table.values})
    
    def _analyze_conversion_funnel(self, events: pd.DataFrame, customer_id_col: str, event_col: str) -> Dict[str, Any]:
        """Analyze conversion funnel."""
        funnel_stages = events[event_col].value_counts()
        
        # Calculate conversion rates (simplified)
        total_customers = events[customer_id_col].nunique()
        
        return {
            'stage_counts': funnel_stages.to_dict(),
            'customer_coverage': {stage: events[events[event_col] == stage][customer_id_col].nunique() / total_customers 
                                for stage in funnel_stages.index}
        }
    
    def _calculate_journey_metrics(self, events: pd.DataFrame, customer_id_col: str, timestamp_col: str) -> Dict[str, float]:
        """Calculate journey metrics."""
        journey_metrics = {}
        
        for customer_id in events[customer_id_col].unique():
            customer_events = events[events[customer_id_col] == customer_id].sort_values(timestamp_col)
            
            if len(customer_events) > 1:
                journey_duration = (customer_events[timestamp_col].max() - customer_events[timestamp_col].min()).days
                journey_events = len(customer_events)
                
                if 'journey_duration' not in journey_metrics:
                    journey_metrics['journey_duration'] = []
                    journey_metrics['journey_events'] = []
                
                journey_metrics['journey_duration'].append(journey_duration)
                journey_metrics['journey_events'].append(journey_events)
        
        if journey_metrics:
            return {
                'avg_journey_duration_days': np.mean(journey_metrics['journey_duration']),
                'avg_journey_events': np.mean(journey_metrics['journey_events']),
                'median_journey_duration_days': np.median(journey_metrics['journey_duration']),
                'median_journey_events': np.median(journey_metrics['journey_events'])
            }
        else:
            return {}
    
    def _analyze_journey_timing(self, events: pd.DataFrame, customer_id_col: str, timestamp_col: str) -> Dict[str, Any]:
        """Analyze journey timing patterns."""
        events['hour'] = events[timestamp_col].dt.hour
        events['day_of_week'] = events[timestamp_col].dt.dayofweek
        events['month'] = events[timestamp_col].dt.month
        
        timing_analysis = {
            'peak_hours': events['hour'].value_counts().head(5).to_dict(),
            'peak_days': events['day_of_week'].value_counts().head(3).to_dict(),
            'seasonal_trends': events['month'].value_counts().to_dict()
        }
        
        return timing_analysis
    
    def _analyze_dropoffs(self, events: pd.DataFrame, customer_id_col: str, event_col: str, journey_stages: List[str]) -> Dict[str, Any]:
        """Analyze journey drop-off points."""
        dropoffs = {}
        
        # Count customers at each stage
        stage_customers = {}
        for stage in journey_stages:
            stage_customers[stage] = events[events[event_col] == stage][customer_id_col].nunique()
        
        # Calculate dropoff rates
        for i, stage in enumerate(journey_stages[:-1]):
            current_stage_customers = stage_customers[stage]
            next_stage_customers = stage_customers.get(journey_stages[i+1], 0)
            
            if current_stage_customers > 0:
                dropoff_rate = (current_stage_customers - next_stage_customers) / current_stage_customers
                dropoffs[stage] = {
                    'customers': current_stage_customers,
                    'dropoff_rate': dropoff_rate,
                    'lost_customers': current_stage_customers - next_stage_customers
                }
        
        return dropoffs
    
    def _define_cluster_personas(self, cluster_analysis: pd.DataFrame, features: List[str]) -> Dict[str, str]:
        """Define customer personas based on cluster characteristics."""
        personas = {}
        
        for cluster_id in cluster_analysis.index:
            # Analyze characteristics to define persona
            cluster_data = cluster_analysis.loc[cluster_id]
            
            # Simple persona rules (can be enhanced)
            if features:
                primary_feature = cluster_data.idxmax()
                primary_value = cluster_data.max()
                
                if primary_value > cluster_data.mean() * 1.5:
                    personas[f'cluster_{cluster_id}'] = f"High {primary_feature.replace('_', ' ').title()} Customer"
                elif primary_value < cluster_data.mean() * 0.5:
                    personas[f'cluster_{cluster_id}'] = f"Low {primary_feature.replace('_', ' ').title()} Customer"
                else:
                    personas[f'cluster_{cluster_id}'] = f"Moderate {primary_feature.replace('_', ' ').title()} Customer"
            else:
                personas[f'cluster_{cluster_id}'] = f"Segment {cluster_id}"
        
        return personas
    
    def _calculate_behavioral_importance(self, scaled_data: np.ndarray, clusters: np.ndarray, features: List[str]) -> Dict[str, float]:
        """Calculate behavioral feature importance."""
        importance = {}
        
        for i, feature in enumerate(features):
            # Calculate silhouette score for each feature
            feature_data = scaled_data[:, i].reshape(-1, 1)
            
            if len(set(clusters)) > 1:
                silhouette_feature = silhouette_score(feature_data, clusters)
                importance[feature] = abs(silhouette_feature)
            else:
                importance[feature] = 0
        
        return importance
    
    def _simple_market_basket_analysis(self, transactions: pd.DataFrame, customer_id_col: str, product_col: str, transaction_id_col: str) -> Dict[str, Any]:
        """Simple market basket analysis without mlxtend."""
        # Calculate product co-occurrence
        transaction_products = transactions.groupby(transaction_id_col)[product_col].apply(list)
        
        product_pairs = {}
        for products in transaction_products:
            if len(products) > 1:
                for i, product1 in enumerate(products):
                    for j, product2 in enumerate(products):
                        if i < j:
                            pair = tuple(sorted([product1, product2]))
                            product_pairs[pair] = product_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        top_pairs = sorted(product_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Calculate product popularity
        product_popularity = transactions[product_col].value_counts()
        
        return {
            'top_product_pairs': top_pairs,
            'product_popularity': product_popularity.head(20).to_dict(),
            'total_unique_products': len(product_popularity),
            'total_transactions': transactions[transaction_id_col].nunique()
        }
    
    def _generate_journey_recommendations(self, funnel_analysis: Dict, dropoff_analysis: Dict) -> List[str]:
        """Generate recommendations based on journey analysis."""
        recommendations = []
        
        # Based on dropoff analysis
        if dropoff_analysis:
            max_dropoff_stage = max(dropoff_analysis.keys(), 
                                  key=lambda x: dropoff_analysis[x]['dropoff_rate'])
            recommendations.append(f"Optimize {max_dropoff_stage} stage to reduce {dropoff_analysis[max_dropoff_stage]['dropoff_rate']:.1%} drop-off rate")
        
        # Based on funnel analysis
        if 'stage_counts' in funnel_analysis:
            lowest_stage = min(funnel_analysis['stage_counts'], key=funnel_analysis['stage_counts'].get)
            recommendations.append(f"Improve conversion to {lowest_stage} stage")
        
        # General recommendations
        recommendations.extend([
            "Implement personalized customer journey experiences",
            "Use A/B testing to optimize touchpoints",
            "Create targeted campaigns for high-drop-off stages",
            "Develop retargeting strategies for abandoned journeys"
        ])
        
        return recommendations