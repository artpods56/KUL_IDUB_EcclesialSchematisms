# AI Osrodek - Experiment Findings Summary

## Overview
This document summarizes all experiments, findings, and results from the Jupyter notebooks in this project. The project appears to focus on document AI for analyzing Ecclesiastical Schematisms (church documents) using various machine learning approaches.

---

## 1. Dataset Analysis (EDA Notebook)

### Dataset: Ecclesiastical Schematisms
- **Source**: `artpods56/EcclesialSchematisms` 
- **Total Examples**: 158 documents
- **Total Tokens**: 27,693 tokens across all documents
- **Average Tokens per Example**: 175.3 tokens
- **Token Range**: 0-290 tokens per document

### Key Data Quality Issues Discovered

#### 1. Severe Class Imbalance
- **"O" (Outside) labels**: 26,757 tokens (96.62%) - Overwhelming majority
- **Meaningful entity labels**: Only 3.38% of all tokens
- This extreme imbalance is a major training challenge

#### 2. Label Distribution
**Meaningful Entity Labels** (very sparse):
- `B-dedication`: 159 tokens (0.57%)
- `I-dedication`: 317 tokens (1.14%) - Most common meaningful class
- `B-parish`: 152 tokens (0.55%)
- `I-parish`: 42 tokens (0.15%)
- `B-building_material`: 120 tokens (0.43%)
- `I-building_material`: 21 tokens (0.08%)
- `B-deanery`: 14 tokens (0.05%)
- `I-deanery`: 12 tokens (0.04%)

**Other Labels**:
- `B-page_number`: 87 tokens (0.31%)
- `B-material`: 8 tokens (0.03%)
- `B-parish_dedication`: 2 tokens (0.01%)
- `B-settlement_classification`: 1 token (0.00%)

#### 3. Example Categorization
- **Empty examples**: 10 (6.3%)
- **Only 'O' labels**: 53 examples (33.5%)
- **Only page_number labels**: 43 examples (27.2%)
- **✅ Positive examples with meaningful content**: 52 examples (32.9%)

### Critical Finding
Only **52 out of 158 examples (32.9%)** contain meaningful entity labels beyond page numbers. This severely limits the training data available for the actual NER task.

### Imbalance Analysis
- **Most common meaningful class**: I-dedication (2.59% in positive examples)
- **Least common meaningful class**: Various classes at 0.01%
- **Imbalance ratio**: 317:1 between most and least common meaningful classes

---

## 2. LayoutLMv3 Experiments

### 2.1 Base LayoutLMv3 Fine-tuning (`finetune_layoutlmv3.ipynb`)
*Note: Full content was truncated, but this notebook appears to contain the main fine-tuning experiments*

**Likely Contents Based on File Structure**:
- Fine-tuning LayoutLMv3 on the Ecclesiastical Schematisms dataset
- Training configuration and hyperparameters
- Training metrics and validation results
- Model performance evaluation

### 2.2 Focal Loss LayoutLMv3 (`focal_finetune_layoutlmv3.ipynb`)
*Note: Content was truncated*

**Purpose**: 
- Addressing the severe class imbalance discovered in EDA
- Using focal loss to better handle the 96.62% "O" label dominance
- Likely improved performance on minority classes

### 2.3 LayoutLMv3 Evaluation (`evaluate_layoutlmv3.ipynb`)
*Note: Content was truncated*

**Likely Contents**:
- Model performance metrics (precision, recall, F1)
- Per-class performance analysis
- Confusion matrices
- Error analysis

### 2.4 LayoutLMv3 Inference (`layoutlmv3_inference.ipynb`)
*Note: Content was truncated*

**Likely Contents**:
- Inference pipeline setup
- Model deployment code
- Real-world document processing examples
- Output visualization

---

## 3. Alternative Approaches

### 3.1 Donut Model Experiments (`donut_finetune.ipynb`)
*Note: Content was truncated*

**Purpose**:
- Testing Donut (Document Understanding Transformer) as an alternative to LayoutLMv3
- Donut is an end-to-end document understanding model
- May handle document layout better than token-based approaches

### 3.2 LLM Annotation (`llm_annotating.ipynb`)
*Note: Content was truncated*

**Purpose**:
- Using Large Language Models for document annotation
- Potentially generating additional training data
- Semi-supervised learning approach
- May help with the data scarcity issue identified in EDA

---

## 4. Data Management and Annotation

### 4.1 Label Studio Integration (`labelstudio_client.ipynb`)
*Note: Content was truncated*

**Purpose**:
- Setting up Label Studio for manual annotation
- Creating annotation workflows
- Quality control for annotations
- Addressing the limited positive examples issue

### 4.2 Geospatial Data Processing (`shapefile_stuff.ipynb`)
*Note: Content was truncated*

**Purpose**:
- Processing geographical data related to church locations
- Potentially enriching the dataset with spatial information
- May be used for validation or additional features

---

## 5. Key Challenges Identified

### 5.1 Data Quality Issues
1. **Severe Class Imbalance**: 96.62% "O" labels vs 3.38% meaningful entities
2. **Limited Training Data**: Only 52/158 examples have meaningful content
3. **High Variance in Class Frequency**: 317:1 ratio between most/least common classes

### 5.2 Training Challenges
1. **Model Bias**: Models likely to predict "O" for everything
2. **Poor Minority Class Performance**: Rare entities hard to learn
3. **Evaluation Complexity**: Standard metrics may be misleading

### 5.3 Domain-Specific Challenges
1. **Historical Documents**: Old formatting and language
2. **Complex Layout**: Church documents have varied structures
3. **OCR Quality**: Text extraction may introduce errors

---

## 6. Proposed Solutions and Approaches

### 6.1 Addressing Class Imbalance
- **Focal Loss**: Implemented in `focal_finetune_layoutlmv3.ipynb`
- **Data Filtering**: Using only positive examples for training
- **Class Weights**: Adjusting loss function weights

### 6.2 Data Augmentation
- **LLM Annotation**: Generating additional training examples
- **Label Studio**: Manual annotation of more examples
- **Synthetic Data**: Potentially creating artificial examples

### 6.3 Model Architecture
- **LayoutLMv3**: Multi-modal approach with text, layout, and visual features
- **Donut**: End-to-end transformer for document understanding
- **Ensemble Methods**: Combining multiple approaches

---

## 7. Expected Results and Outcomes

### 7.1 Performance Expectations
Given the data quality issues:
- **Overall Accuracy**: Likely high due to "O" label dominance
- **Entity Recognition**: Challenging due to data scarcity
- **Recall vs Precision Trade-off**: Focus on recall for rare entities

### 7.2 Success Metrics
- **F1 Score per Entity Type**: More meaningful than overall accuracy
- **Entity-Level Evaluation**: Complete entity extraction success
- **Domain Expert Validation**: Human evaluation of practical utility

---

## 8. Recommendations

### 8.1 Immediate Actions
1. **Review Training Results**: Analyze performance metrics from fine-tuning experiments
2. **Error Analysis**: Identify specific failure patterns
3. **Data Collection**: Prioritize annotation of more positive examples

### 8.2 Long-term Improvements
1. **Active Learning**: Use model uncertainty to guide annotation
2. **Multi-task Learning**: Combine related tasks to improve data efficiency
3. **Transfer Learning**: Leverage models trained on similar document types

### 8.3 Evaluation Strategy
1. **Stratified Evaluation**: Separate metrics for different document types
2. **Error Categories**: Classify and prioritize different error types
3. **User Studies**: Validate practical utility with domain experts

---

## 9. Technical Implementation

### 9.1 Model Architecture Choices
- **LayoutLMv3**: Chosen for multi-modal document understanding
- **Focal Loss**: Addressing class imbalance
- **Fine-tuning Strategy**: Domain-specific adaptation

### 9.2 Data Processing Pipeline
1. **OCR Text Extraction**: Converting images to text
2. **Bounding Box Detection**: Layout information
3. **Token Classification**: NER task setup
4. **Post-processing**: Entity assembly and validation

---

## 10. Conclusions

### Key Findings
1. **Data Quality is Critical**: Only 32.9% of examples contain meaningful entities
2. **Class Imbalance is Severe**: 96.62% "O" labels create training challenges
3. **Multiple Approaches Needed**: No single solution addresses all challenges

### Success Factors
1. **Data Quality Improvement**: More positive examples needed
2. **Specialized Training**: Focal loss and class balancing essential
3. **Domain Expertise**: Human validation crucial for practical success

### Future Directions
1. **Expand Dataset**: More annotated examples needed
2. **Ensemble Methods**: Combine different model approaches
3. **Production Pipeline**: Focus on practical deployment considerations

---

*Note: Some specific results and metrics were not accessible due to truncated notebook content. This summary is based on the available information and inferred from the experimental setup and data analysis findings.*
