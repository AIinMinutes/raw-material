# Cross-Validation Techniques - Notes

## 1. K-Fold Cross-Validation

**Basic Idea:**
- Split data into k equal-sized folds
- Use k-1 folds for training, 1 fold for validation
- Repeat k times, using each fold once as validation

**Why:**
- Ensures all data points are used for both training and validation
- Reduces variance in performance estimation

**Algorithm:**
1. Divide dataset into k equal parts
2. For i = 1 to k:
   - Use fold i as validation set
   - Use remaining k-1 folds as training set
   - Train model and record performance
3. Average the k performance metrics

**Pros:**
- Uses all data for both training and validation
- More reliable estimate than simple train-test split
- Works well with medium-sized datasets

**Cons:**
- Not optimal for imbalanced datasets without stratification
- Computationally expensive for large datasets
- Doesn't account for time dependencies

**Python Implementation:**
```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

## 2. Hold-Out Cross-Validation

**Basic Idea:**
- Split data into training and test sets once (typically 70-30%)
- Train on training set, evaluate on test set

**Why:**
- Simple, fast approach to model evaluation
- Suitable when data is abundant

**Algorithm:**
1. Randomly select x% (typically 70%) of data for training
2. Use remaining (100-x)% for testing
3. Train model on training set, evaluate on test set

**Pros:**
- Computationally efficient
- Simple to implement
- Good for very large datasets

**Cons:**
- High variance in performance estimation
- Inefficient use of data
- Test set may not be representative

**Python Implementation:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

## 3. Stratified K-Fold Cross-Validation

**Basic Idea:**
- Similar to k-fold but preserves class distribution in each fold
- Each fold has same proportion of each class as in full dataset

**Why:**
- Ensures representative sampling for imbalanced datasets
- Reduces bias in performance estimation

**Algorithm:**
1. Divide dataset into k folds, maintaining class proportions in each fold
2. For i = 1 to k:
   - Use fold i as validation set
   - Use remaining k-1 folds as training set
   - Train model and record performance
3. Average the k performance metrics

**Pros:**
- Works well with imbalanced datasets
- More reliable than standard k-fold for classification
- Reduces bias in performance estimation

**Cons:**
- Slightly more complex implementation
- Not suitable for time series data
- Still computationally intensive for large datasets

**Python Implementation:**
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

## 4. Leave-P-Out Cross-Validation

**Basic Idea:**
- Use p samples for validation, n-p for training
- Test all possible combinations of p samples

**Why:**
- Exhaustive testing of all possible train-test splits
- Theoretically most thorough validation approach

**Algorithm:**
1. For each possible combination of p samples from n total samples:
   - Use those p samples as validation set
   - Use remaining n-p samples as training set
   - Train model and record performance
2. Average all performance metrics

**Pros:**
- Exhaustive evaluation of model performance
- Theoretically most thorough approach
- No randomness in splitting

**Cons:**
- Computationally infeasible for large datasets
- Number of combinations grows exponentially: C(n,p)
- Impractical for most real-world applications

**Python Implementation:**
```python
from sklearn.model_selection import LeavePOut
lpo = LeavePOut(p=2)
for train_index, test_index in lpo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

## 5. Leave-One-Out Cross-Validation

**Basic Idea:**
- Special case of Leave-P-Out with p=1
- Use one sample for validation, all others for training
- Repeat for each sample

**Why:**
- Maximizes training data while still testing on all samples
- Deterministic process with no random component

**Algorithm:**
1. For i = 1 to n (where n is number of samples):
   - Use sample i as validation set
   - Use all other samples as training set
   - Train model and record performance
2. Average all n performance metrics

**Pros:**
- Uses maximum amount of data for training
- No randomness in results
- Good for small datasets

**Cons:**
- Computationally expensive (n iterations)
- High variance in individual fold results
- Not suitable for very large datasets

**Python Implementation:**
```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

## 6. Monte Carlo Cross-Validation

**Basic Idea:**
- Randomly split data into training/test sets multiple times
- Same split ratio each time, but different random samples

**Why:**
- Reduces dependency on a single split
- Provides distribution of performance metrics

**Algorithm:**
1. For i = 1 to m (number of iterations):
   - Randomly split data into training and test sets (fixed ratio)
   - Train model on training set, evaluate on test set
   - Record performance
2. Average all m performance metrics

**Pros:**
- Flexible number of iterations
- Less computationally intensive than exhaustive methods
- Provides performance distribution statistics

**Cons:**
- Some samples might never be in test set
- Others might be in test set multiple times
- Non-deterministic results

**Python Implementation:**
```python
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
for train_index, test_index in ss.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

## 7. Time Series Cross-Validation

**Basic Idea:**
- Respects temporal order of data
- Training set always precedes test set chronologically

**Why:**
- Prevents data leakage in time-dependent data
- Mimics real-world forecasting scenarios

**Algorithm:**
1. Start with small initial training set
2. For each time step:
   - Train on all data up to time t
   - Test on data at time t+1 (or t+1 to t+h)
   - Expand training window and repeat

**Pros:**
- Respects temporal dependencies
- Simulates real-world forecasting scenarios
- Prevents future data from leaking into model training

**Cons:**
- Initial models trained on less data
- Requires sufficient time series length
- More complex implementation

**Python Implementation:**
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

## 8. Nested Cross-Validation

**Basic Idea:**
- Two-level cross-validation: outer CV for performance estimation, inner CV for hyperparameter tuning
- Separates model selection from model evaluation

**Why:**
- Provides unbiased performance estimation while optimizing hyperparameters
- Prevents overfitting to validation data

**Algorithm:**
1. Split data into k outer folds
2. For each outer fold:
   - Split remaining k-1 folds into j inner folds
   - Use inner folds to tune hyperparameters
   - Evaluate final model on outer fold
3. Average outer fold performance metrics

**Pros:**
- Reduces bias in performance estimation
- Properly separates model selection from evaluation
- More reliable for comparing different algorithms

**Cons:**
- Computationally expensive (k × j iterations)
- Complex implementation
- Requires substantial amount of data

**Python Implementation:**
```python
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

# Outer cross-validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []

for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner cross-validation for hyperparameter tuning
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    model = LogisticRegression()
    param_grid = {'C': [0.1, 1.0, 10.0]}
    
    clf = GridSearchCV(model, param_grid, cv=inner_cv)
    clf.fit(X_train, y_train)
    
    # Evaluate on outer test fold
    score = clf.score(X_test, y_test)
    outer_scores.append(score)

print(f"Nested CV Score: {np.mean(outer_scores):.4f}")
```

## 9. Group K-Fold Cross-Validation

**Basic Idea:**
- Similar to k-fold but respects group structure in data
- Ensures samples from same group never span train and test sets

**Why:**
- Prevents data leakage when samples have dependencies
- Critical for datasets with grouped observations (e.g., multiple samples from same patient)

**Algorithm:**
1. Identify groups in the dataset
2. Create k folds ensuring samples from same group are in same fold
3. For i = 1 to k:
   - Use fold i as validation set
   - Use remaining k-1 folds as training set
   - Train model and record performance
4. Average the k performance metrics

**Pros:**
- Prevents information leakage between related samples
- Provides realistic performance estimates for grouped data
- Essential for medical, sensor, or user-level data

**Cons:**
- Requires group labels for all samples
- May result in imbalanced fold sizes
- Less flexibility in fold construction

**Python Implementation:**
```python
from sklearn.model_selection import GroupKFold
# Assume groups is an array specifying group membership for each sample
gkf = GroupKFold(n_splits=5)
for train_index, test_index in gkf.split(X, y, groups=groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

## 10. Repeated K-Fold Cross-Validation

**Basic Idea:**
- Run k-fold cross-validation multiple times with different random splits
- Average results across all repetitions

**Why:**
- Reduces variance in performance estimation
- Provides more robust performance metrics

**Algorithm:**
1. Repeat r times:
   - Perform k-fold cross-validation with different random splits
   - Record performance metrics for each fold
2. Average all r × k performance metrics

**Pros:**
- More stable performance estimates
- Reduces impact of "lucky" or "unlucky" splits
- Better statistical properties

**Cons:**
- More computationally expensive than single k-fold
- Diminishing returns after certain number of repetitions
- Still may not handle imbalanced data well without stratification

**Python Implementation:**
```python
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

## 11. Blocked Cross-Validation

**Basic Idea:**
- Split data into blocks based on a blocking factor
- Ensure training and test sets contain different blocks

**Why:**
- Respects spatial, temporal, or other structured dependencies
- Prevents overly optimistic performance estimates

**Algorithm:**
1. Divide data into blocks based on blocking factor
2. For each fold:
   - Select some blocks for testing
   - Use remaining blocks for training
   - Train model and record performance
3. Average performance metrics across all folds

**Pros:**
- Handles spatial or temporal autocorrelation
- More realistic for geospatial or environmental data
- Prevents information leakage between related observations

**Cons:**
- Requires domain knowledge to identify blocking factors
- May result in irregular fold sizes
- Not directly implemented in scikit-learn

**Python Implementation:**
```python
# Custom implementation example (not from scikit-learn)
def blocked_cv(X, y, blocks, n_splits=5):
    unique_blocks = np.unique(blocks)
    np.random.shuffle(unique_blocks)
    
    # Split blocks into n_splits
    block_folds = np.array_split(unique_blocks, n_splits)
    
    for i in range(n_splits):
        test_blocks = block_folds[i]
        test_idx = np.where(np.isin(blocks, test_blocks))[0]
        train_idx = np.where(~np.isin(blocks, test_blocks))[0]
        
        yield train_idx, test_idx
```

## 12. Bootstrapping

**Basic Idea:**
- Create training sets by random sampling with replacement
- Use out-of-bag (unselected) samples for validation

**Why:**
- Provides robust error estimates
- Works well with small datasets

**Algorithm:**
1. For i = 1 to B (number of bootstrap samples):
   - Create bootstrap sample by randomly sampling n samples with replacement
   - Train model on bootstrap sample
   - Test on samples not selected (out-of-bag)
   - Record performance
2. Average all B performance metrics

**Pros:**
- Works well for small datasets
- Provides confidence intervals for model performance
- Foundation for ensemble methods like Random Forests

**Cons:**
- Out-of-bag samples not fixed in size
- More complex than standard cross-validation
- May overestimate performance in some cases

**Python Implementation:**
```python
from sklearn.utils import resample

def bootstrap_cv(X, y, n_iterations=100, model=None):
    n_samples = X.shape[0]
    scores = []
    
    for i in range(n_iterations):
        # Generate bootstrap indices (with replacement)
        indices = resample(range(n_samples), replace=True, n_samples=n_samples)
        # Out-of-bag indices
        oob_indices = [i for i in range(n_samples) if i not in indices]
        
        # Split data
        X_train, X_test = X[indices], X[oob_indices]
        y_train, y_test = y[indices], y[oob_indices]
        
        # Train and evaluate
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```