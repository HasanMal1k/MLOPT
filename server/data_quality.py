"""
Data Quality Assessment and Model Recommendation System

This module calculates data quality scores and recommends ML algorithms
based on actual dataset characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Analyzes dataset quality and characteristics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_rows = len(df)
        self.n_cols = len(df.columns)
        
    def calculate_quality_score(self) -> Dict[str, Any]:
        """
        Calculate comprehensive quality score (0-100) based on multiple factors
        
        Returns:
            Dict containing quality_score, quality_rating, and breakdown
        """
        scores = {}
        weights = {}
        
        # 1. Completeness Score (30% weight) - Missing data assessment
        completeness = self._calculate_completeness()
        scores['completeness'] = completeness
        weights['completeness'] = 0.30
        
        # 2. Size Score (20% weight) - Dataset size adequacy
        size_score = self._calculate_size_score()
        scores['size'] = size_score
        weights['size'] = 0.20
        
        # 3. Feature Quality Score (25% weight) - Feature diversity and quality
        feature_quality = self._calculate_feature_quality()
        scores['feature_quality'] = feature_quality
        weights['feature_quality'] = 0.25
        
        # 4. Data Validity Score (15% weight) - Data type consistency
        validity = self._calculate_validity()
        scores['validity'] = validity
        weights['validity'] = 0.15
        
        # 5. Balance Score (10% weight) - Distribution balance
        balance = self._calculate_balance()
        scores['balance'] = balance
        weights['balance'] = 0.10
        
        # Calculate weighted total
        total_score = sum(scores[k] * weights[k] for k in scores.keys())
        total_score = round(total_score, 2)
        
        # Determine rating
        if total_score >= 90:
            rating = "Excellent"
        elif total_score >= 70:
            rating = "Good"
        elif total_score >= 50:
            rating = "Fair"
        else:
            rating = "Poor"
            
        return {
            "quality_score": total_score,
            "quality_rating": rating,
            "score_breakdown": scores,
            "weights": weights
        }
    
    def _calculate_completeness(self) -> float:
        """Calculate completeness score based on missing data"""
        missing_ratio = self.df.isnull().sum().sum() / (self.n_rows * self.n_cols)
        
        if missing_ratio == 0:
            return 100
        elif missing_ratio < 0.05:
            return 95
        elif missing_ratio < 0.10:
            return 85
        elif missing_ratio < 0.20:
            return 70
        elif missing_ratio < 0.30:
            return 50
        else:
            return max(20, 100 - (missing_ratio * 200))
    
    def _calculate_size_score(self) -> float:
        """Calculate score based on dataset size"""
        # Optimal: 1000+ rows, 3+ features
        row_score = min(100, (self.n_rows / 1000) * 100)
        
        if self.n_cols < 2:
            col_score = 30
        elif self.n_cols < 3:
            col_score = 60
        elif self.n_cols < 5:
            col_score = 80
        else:
            col_score = 100
            
        return (row_score * 0.7) + (col_score * 0.3)
    
    def _calculate_feature_quality(self) -> float:
        """Calculate feature quality score"""
        scores = []
        
        for col in self.df.columns:
            col_score = 100
            
            # Check for single unique value (useless feature)
            if self.df[col].nunique() == 1:
                col_score = 0
            # Check for too many unique values in categorical
            elif self.df[col].dtype == 'object':
                unique_ratio = self.df[col].nunique() / self.n_rows
                if unique_ratio > 0.9:  # Almost all unique (like IDs)
                    col_score = 30
                elif unique_ratio > 0.5:
                    col_score = 60
            # Check for variance in numeric columns
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].std() == 0:  # No variance
                    col_score = 0
                    
            scores.append(col_score)
            
        return np.mean(scores) if scores else 50
    
    def _calculate_validity(self) -> float:
        """Calculate data validity score"""
        validity_scores = []
        
        for col in self.df.columns:
            # Check for data type consistency
            if self.df[col].dtype == 'object':
                # For object columns, check if values are reasonable
                avg_length = self.df[col].dropna().astype(str).str.len().mean()
                if avg_length > 1000:  # Suspiciously long text
                    validity_scores.append(70)
                else:
                    validity_scores.append(100)
            else:
                # For numeric columns, check for outliers
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_ratio = ((self.df[col] < (Q1 - 3 * IQR)) | 
                                    (self.df[col] > (Q3 + 3 * IQR))).sum() / self.n_rows
                    
                    if outlier_ratio > 0.1:
                        validity_scores.append(70)
                    else:
                        validity_scores.append(100)
                else:
                    validity_scores.append(100)
                    
        return np.mean(validity_scores) if validity_scores else 100
    
    def _calculate_balance(self) -> float:
        """Calculate balance score for categorical distributions"""
        # Find potential target columns (categorical with few unique values)
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        balance_scores = []
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            if 2 <= unique_count <= 20:  # Potential target column
                value_counts = self.df[col].value_counts()
                max_ratio = value_counts.max() / len(self.df)
                
                if max_ratio < 0.6:  # Well balanced
                    balance_scores.append(100)
                elif max_ratio < 0.8:  # Moderate imbalance
                    balance_scores.append(70)
                else:  # Highly imbalanced
                    balance_scores.append(40)
                    
        return np.mean(balance_scores) if balance_scores else 90
    
    def analyze_characteristics(self) -> Dict[str, Any]:
        """
        Analyze dataset characteristics for model recommendation
        
        Returns:
            Dict containing data characteristics
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        # Calculate ratios
        numeric_ratio = len(numeric_cols) / self.n_cols if self.n_cols > 0 else 0
        categorical_ratio = len(categorical_cols) / self.n_cols if self.n_cols > 0 else 0
        
        # Missing data ratio
        missing_ratio = self.df.isnull().sum().sum() / (self.n_rows * self.n_cols)
        
        # Potential target columns
        potential_targets = []
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            if 2 <= unique_count <= 20:
                potential_targets.append({
                    "column": col,
                    "unique_values": int(unique_count),
                    "type": "classification"
                })
                
        for col in numeric_cols:
            if self.df[col].nunique() > 20:  # Continuous target
                potential_targets.append({
                    "column": col,
                    "unique_values": int(self.df[col].nunique()),
                    "type": "regression"
                })
        
        # Feature complexity
        high_cardinality_features = []
        for col in categorical_cols:
            if self.df[col].nunique() > 50:
                high_cardinality_features.append(col)
        
        return {
            "numeric_ratio": round(numeric_ratio, 3),
            "categorical_ratio": round(categorical_ratio, 3),
            "missing_ratio": round(missing_ratio, 3),
            "feature_count": self.n_cols,
            "sample_count": self.n_rows,
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols),
            "potential_targets": potential_targets[:5],  # Top 5
            "high_cardinality_features": high_cardinality_features[:5],
            "is_time_series_candidate": self._detect_time_series(),
            "has_text_data": any(self.df[col].astype(str).str.len().mean() > 100 
                                 for col in categorical_cols if col in self.df.columns)
        }
    
    def _detect_time_series(self) -> bool:
        """Detect if dataset is likely time series"""
        # Look for date/datetime columns
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                return True
            # Try to parse as datetime
            try:
                pd.to_datetime(self.df[col], errors='coerce')
                if self.df[col].notna().sum() / len(self.df) > 0.5:
                    return True
            except:
                continue
        return False


class ModelRecommender:
    """Recommends ML algorithms based on dataset characteristics"""
    
    # Model suitability definitions
    CLASSIFICATION_MODELS = {
        "Logistic Regression": {
            "best_for": ["small", "linear", "binary"],
            "requirements": {"min_samples": 100, "numeric_heavy": True}
        },
        "Random Forest": {
            "best_for": ["medium", "nonlinear", "multiclass", "robust"],
            "requirements": {"min_samples": 200}
        },
        "XGBoost": {
            "best_for": ["large", "nonlinear", "high_performance"],
            "requirements": {"min_samples": 500}
        },
        "Gradient Boosting": {
            "best_for": ["medium", "nonlinear", "high_performance"],
            "requirements": {"min_samples": 300}
        },
        "Decision Tree": {
            "best_for": ["small", "interpretable"],
            "requirements": {"min_samples": 50}
        },
        "SVM": {
            "best_for": ["small", "linear", "high_dimensional"],
            "requirements": {"min_samples": 100, "numeric_heavy": True}
        },
        "K-Nearest Neighbors": {
            "best_for": ["small", "nonlinear"],
            "requirements": {"min_samples": 100, "numeric_heavy": True}
        },
        "Naive Bayes": {
            "best_for": ["small", "categorical_heavy", "text"],
            "requirements": {"min_samples": 50}
        },
        "Neural Network": {
            "best_for": ["large", "complex", "nonlinear"],
            "requirements": {"min_samples": 1000}
        }
    }
    
    REGRESSION_MODELS = {
        "Linear Regression": {
            "best_for": ["small", "linear"],
            "requirements": {"min_samples": 100, "numeric_heavy": True}
        },
        "Random Forest": {
            "best_for": ["medium", "nonlinear", "robust"],
            "requirements": {"min_samples": 200}
        },
        "XGBoost": {
            "best_for": ["large", "nonlinear", "high_performance"],
            "requirements": {"min_samples": 500}
        },
        "Gradient Boosting": {
            "best_for": ["medium", "nonlinear"],
            "requirements": {"min_samples": 300}
        },
        "Decision Tree": {
            "best_for": ["small", "interpretable"],
            "requirements": {"min_samples": 50}
        },
        "SVR": {
            "best_for": ["small", "nonlinear"],
            "requirements": {"min_samples": 100, "numeric_heavy": True}
        },
        "Ridge/Lasso": {
            "best_for": ["small", "linear", "regularization"],
            "requirements": {"min_samples": 100, "numeric_heavy": True}
        }
    }
    
    TIME_SERIES_MODELS = ["ARIMA", "SARIMA", "Prophet", "LSTM", "Linear Regression", "XGBoost"]
    
    def __init__(self, characteristics: Dict[str, Any]):
        self.chars = characteristics
        
    def recommend_models(self, task_type: str = "auto", top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend top N models for the dataset
        
        Args:
            task_type: "classification", "regression", "time_series", or "auto"
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended models with reasons
        """
        # Auto-detect task type if needed
        if task_type == "auto":
            task_type = self._detect_task_type()
        
        if self.chars.get("is_time_series_candidate"):
            return self._recommend_time_series_models(top_n)
        elif task_type == "classification":
            return self._recommend_classification_models(top_n)
        elif task_type == "regression":
            return self._recommend_regression_models(top_n)
        else:
            # Return general recommendations
            return self._recommend_general_models(top_n)
    
    def _detect_task_type(self) -> str:
        """Auto-detect the most likely task type"""
        if self.chars.get("is_time_series_candidate"):
            return "time_series"
        
        # Look at potential targets
        targets = self.chars.get("potential_targets", [])
        if not targets:
            return "regression"  # Default
        
        # Count classification vs regression targets
        classification_count = sum(1 for t in targets if t.get("type") == "classification")
        regression_count = sum(1 for t in targets if t.get("type") == "regression")
        
        return "classification" if classification_count >= regression_count else "regression"
    
    def _recommend_classification_models(self, top_n: int) -> List[Dict[str, Any]]:
        """Recommend classification models"""
        recommendations = []
        n_samples = self.chars["sample_count"]
        numeric_ratio = self.chars["numeric_ratio"]
        n_features = self.chars["feature_count"]
        
        for model, props in self.CLASSIFICATION_MODELS.items():
            score = 0
            reasons = []
            
            # Check sample size
            min_samples = props["requirements"].get("min_samples", 0)
            if n_samples >= min_samples:
                score += 20
            else:
                score += max(0, (n_samples / min_samples) * 20)
            
            # Check numeric ratio
            if props["requirements"].get("numeric_heavy") and numeric_ratio > 0.5:
                score += 15
                reasons.append("Suitable for numeric features")
            
            # Dataset size preference
            if n_samples < 300 and "small" in props["best_for"]:
                score += 25
                reasons.append("Optimized for small datasets")
            elif 300 <= n_samples < 1000 and "medium" in props["best_for"]:
                score += 25
                reasons.append("Ideal for medium datasets")
            elif n_samples >= 1000 and "large" in props["best_for"]:
                score += 25
                reasons.append("Excellent for large datasets")
            
            # Performance characteristics
            if "high_performance" in props["best_for"]:
                score += 20
                reasons.append("High performance algorithm")
            
            if "robust" in props["best_for"]:
                score += 10
                reasons.append("Robust to outliers")
            
            if "interpretable" in props["best_for"]:
                score += 10
                reasons.append("Highly interpretable")
            
            recommendations.append({
                "model": model,
                "score": min(100, score),
                "reasons": reasons,
                "category": "classification"
            })
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_n]
    
    def _recommend_regression_models(self, top_n: int) -> List[Dict[str, Any]]:
        """Recommend regression models"""
        recommendations = []
        n_samples = self.chars["sample_count"]
        numeric_ratio = self.chars["numeric_ratio"]
        
        for model, props in self.REGRESSION_MODELS.items():
            score = 0
            reasons = []
            
            min_samples = props["requirements"].get("min_samples", 0)
            if n_samples >= min_samples:
                score += 20
            else:
                score += max(0, (n_samples / min_samples) * 20)
            
            if props["requirements"].get("numeric_heavy") and numeric_ratio > 0.5:
                score += 15
                reasons.append("Suitable for numeric features")
            
            if n_samples < 300 and "small" in props["best_for"]:
                score += 25
                reasons.append("Optimized for small datasets")
            elif 300 <= n_samples < 1000 and "medium" in props["best_for"]:
                score += 25
                reasons.append("Ideal for medium datasets")
            elif n_samples >= 1000 and "large" in props["best_for"]:
                score += 25
                reasons.append("Excellent for large datasets")
            
            if "high_performance" in props["best_for"]:
                score += 20
                reasons.append("High performance algorithm")
            
            if "robust" in props["best_for"]:
                score += 10
                reasons.append("Robust to outliers")
            
            recommendations.append({
                "model": model,
                "score": min(100, score),
                "reasons": reasons,
                "category": "regression"
            })
        
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_n]
    
    def _recommend_time_series_models(self, top_n: int) -> List[Dict[str, Any]]:
        """Recommend time series models"""
        n_samples = self.chars["sample_count"]
        
        recommendations = [
            {
                "model": "Prophet",
                "score": 95 if n_samples > 100 else 70,
                "reasons": ["Excellent for time series with seasonality", "Handles missing data well"],
                "category": "time_series"
            },
            {
                "model": "ARIMA",
                "score": 90 if n_samples > 50 else 60,
                "reasons": ["Classical time series method", "Good for stationary data"],
                "category": "time_series"
            },
            {
                "model": "XGBoost",
                "score": 85 if n_samples > 300 else 50,
                "reasons": ["High performance", "Can capture complex patterns"],
                "category": "time_series"
            },
            {
                "model": "LSTM",
                "score": 80 if n_samples > 1000 else 40,
                "reasons": ["Deep learning approach", "Captures long-term dependencies"],
                "category": "time_series"
            },
            {
                "model": "Linear Regression",
                "score": 70,
                "reasons": ["Simple baseline", "Fast training"],
                "category": "time_series"
            }
        ]
        
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_n]
    
    def _recommend_general_models(self, top_n: int) -> List[Dict[str, Any]]:
        """General recommendations when task type is unclear"""
        return [
            {"model": "Random Forest", "score": 90, "reasons": ["Versatile", "Good for most tasks"], "category": "general"},
            {"model": "XGBoost", "score": 85, "reasons": ["High performance", "Industry standard"], "category": "general"},
            {"model": "Gradient Boosting", "score": 80, "reasons": ["Powerful ensemble method"], "category": "general"},
            {"model": "Linear Models", "score": 70, "reasons": ["Fast", "Interpretable"], "category": "general"},
            {"model": "Decision Tree", "score": 65, "reasons": ["Simple", "Interpretable"], "category": "general"}
        ][:top_n]


def analyze_dataset(df: pd.DataFrame, task_type: str = "auto") -> Dict[str, Any]:
    """
    Complete dataset analysis including quality and model recommendations
    
    Args:
        df: pandas DataFrame
        task_type: Task type ("classification", "regression", "time_series", or "auto")
    
    Returns:
        Dict containing quality_score, quality_rating, characteristics, and recommended_models
    """
    try:
        # Quality analysis
        quality_analyzer = DataQualityAnalyzer(df)
        quality_results = quality_analyzer.calculate_quality_score()
        characteristics = quality_analyzer.analyze_characteristics()
        
        # Model recommendations
        recommender = ModelRecommender(characteristics)
        recommended_models = recommender.recommend_models(task_type=task_type, top_n=5)
        
        return {
            "quality_score": quality_results["quality_score"],
            "quality_rating": quality_results["quality_rating"],
            "score_breakdown": quality_results["score_breakdown"],
            "data_characteristics": characteristics,
            "recommended_models": recommended_models
        }
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return {
            "quality_score": 50.0,
            "quality_rating": "Fair",
            "score_breakdown": {},
            "data_characteristics": {},
            "recommended_models": []
        }
