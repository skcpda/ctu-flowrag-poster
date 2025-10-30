# 📊 Comprehensive Statistics for Research Paper

**Dataset**: CTU FlowRAG - Government Scheme Relation Graphs  
**Generated**: October 2, 2025  
**Status**: Production-Ready, Frozen Dataset

---

## 🎯 **Executive Summary**

This document provides comprehensive statistics for the CTU FlowRAG dataset, a large-scale collection of 3,048 government scheme documents processed into Content Thematic Unit (CTU) relation graphs. The dataset contains 187,220 CTUs connected by 191,128 relations, with intelligent section detection and role-based semantic understanding.

---

## 📈 **Dataset Overview**

| Metric | Value |
|--------|-------|
| **Total Documents** | 3,048 |
| **Total CTUs** | 187,220 |
| **Total Relations** | 191,128 |
| **Structural Relations** | 170,483 (89.2%) |
| **Semantic Relations** | 20,645 (10.8%) |
| **Average CTUs per Document** | 61.4 |
| **Average Relations per Document** | 62.7 |
| **Average Sections per Document** | 5.5 |
| **Maximum Sections per Document** | 9 |

---

## 🔗 **Relation Type Distribution**

| Relation Type | Count | Percentage | Description |
|---------------|-------|------------|-------------|
| **PRECEDES** | 170,483 | 89.2% | Sequential flow within sections |
| **PREREQUISITE_OF** | 11,656 | 6.1% | Prerequisite relationships |
| **ADMINISTERED_BY** | 5,638 | 3.0% | Administrative authority |
| **CAP_LIMITS** | 1,688 | 0.9% | Financial/benefit limits |
| **ENABLES** | 1,663 | 0.9% | Enabling relationships |
| **ELABORATES** | 0 | 0.0% | Elaboration relationships (folded into structural) |
| **SUPPORTS** | 0 | 0.0% | Support relationships (folded into structural) |

### **Relation Characteristics**
- **Structural Relations**: 89.2% (PRECEDES, SEGMENT_CONTINUATION)
- **Semantic Relations**: 10.8% (PREREQUISITE_OF, ADMINISTERED_BY, CAP_LIMITS, ENABLES)
- **ELABORATES/SUPPORTS**: Folded into structural relations (SEGMENT_CONTINUATION)
- **Method Distribution**: 100% production_pipeline (consistent processing)

---

## 👥 **Role Distribution**

| Role | Count | Percentage | Description |
|------|-------|------------|-------------|
| **ContextObjective** | 75,892 | 40.5% | Scheme context and objectives |
| **BenefitsAssistance** | 48,526 | 25.9% | Benefits and assistance details |
| **Eligibility** | 21,564 | 11.5% | Eligibility criteria |
| **AuthoritiesGovernance** | 16,887 | 9.0% | Administrative authorities |
| **ApplicationProcess** | 15,848 | 8.5% | Application procedures |
| **TimelineFrequency** | 7,473 | 4.0% | Timeline and frequency info |
| **DefinitionsReferences** | 1,030 | 0.6% | Definitions and references |

### **Role Coverage**
- **Unique Roles**: 7 distinct semantic roles
- **Role Coverage**: 100% of documents contain all major role types
- **Most Common**: ContextObjective (40.5%) - scheme introductions and objectives
- **Least Common**: DefinitionsReferences (0.6%) - technical definitions

---

## 🏗️ **Graph Structure Analysis**

### **Density Metrics**
- **Average Density**: 0.017139 ± 0.002288
- **Density Range**: 0.010 - 0.025 (controlled sparsity)
- **Graph Type**: Sparse, well-controlled connectivity

### **Semantic Ratio**
- **Average Semantic Ratio**: 0.108 ± 0.038
- **Semantic Ratio Range**: 0.05 - 0.20
- **Quality Distribution**:
  - High semantic ratio (>0.15): 414 files (13.6%)
  - Medium semantic ratio (0.05-0.15): 2,465 files (80.9%)
  - Low semantic ratio (<0.05): 166 files (5.4%)

### **Adjacency Completeness**
- **PRECEDES Coverage**: 100% complete within sections
- **Cross-Section PRECEDES**: Intentionally disallowed (section-bounded design)
- **Structural Flow**: Complete sequential flow within each section
- **Semantic Cross-Section**: Allowed for meaningful relationships

---

## 📑 **Section Analysis**

| Metric | Value |
|--------|-------|
| **Single Section Files** | 4 (0.1%) |
| **Multi-Section Files** | 3,044 (99.9%) |
| **Maximum Sections per File** | 9 |
| **Average Sections per File** | 5.5 ± 1.3 |
| **Most Common Section Count** | 6 sections (934 files) |

### **Section Distribution**
- **2 sections**: 11 files
- **3 sections**: 202 files
- **4 sections**: 435 files
- **5 sections**: 786 files
- **6 sections**: 934 files (most common)
- **7 sections**: 556 files
- **8 sections**: 114 files
- **9 sections**: 3 files

---

## 🎯 **Confidence Analysis**

### **Raw Confidence Statistics**
- **Mean**: 0.934 ± 0.050
- **Range**: 0.700 - 0.950
- **Distribution**: All high confidence (0.7-1.0)

### **Calibrated Confidence Statistics**
- **Mean**: 0.748 ± 0.040
- **Range**: 0.560 - 0.760
- **Distribution**:
  - High (0.7-1.0): 170,483 edges (structural)
  - Medium (0.3-0.7): 20,645 edges (semantic)

### **Confidence Calibration**
- **Structural Edges**: High confidence (0.7-1.0) - reliable sequential flow
- **Semantic Edges**: Medium confidence (0.3-0.7) - probabilistic relationships
- **Calibration Method**: Method-aware confidence scaling

---

## 📊 **Document Analysis**

### **Document Length Statistics**
- **Mean CTUs per Document**: 61.4
- **Standard Deviation**: 15.2
- **Range**: 12 - 156 CTUs
- **Median**: 58 CTUs

### **Document Complexity**
- **Average Relations per Document**: 62.7
- **Average Sections per Document**: 5.5
- **Document Density**: 0.017139 ± 0.002288

---

## 🔗 **Role Pair Analysis**

### **Top 10 Role Pair Combinations**

| Source Role | Target Role | Count | Description |
|-------------|-------------|-------|-------------|
| ContextObjective | ContextObjective | 34,208 | Narrative flow within objectives |
| ContextObjective | BenefitsAssistance | 21,551 | Objectives to benefits |
| BenefitsAssistance | ContextObjective | 21,337 | Benefits to context |
| BenefitsAssistance | BenefitsAssistance | 16,309 | Benefit elaboration |
| Eligibility | ApplicationProcess | 15,734 | Prerequisite flow |
| Eligibility | Eligibility | 9,271 | Eligibility criteria elaboration |
| AuthoritiesGovernance | ContextObjective | 8,710 | Authority to context |
| ContextObjective | AuthoritiesGovernance | 7,624 | Context to authority |
| AuthoritiesGovernance | BenefitsAssistance | 7,568 | Authority to benefits |
| ApplicationProcess | ApplicationProcess | 6,025 | Process step elaboration |

### **Role Transition Patterns**
- **Most Common**: ContextObjective → ContextObjective (narrative flow)
- **Prerequisite Flow**: Eligibility → ApplicationProcess (logical sequence)
- **Benefit Flow**: ContextObjective → BenefitsAssistance (objective to benefit)
- **Authority Flow**: AuthoritiesGovernance → ContextObjective (authority context)

---

## 🎯 **Quality Metrics**

### **Graph Quality**
- **Sparse Graphs**: 3,045 files (99.9%) - controlled density
- **Dense Graphs**: 0 files - no over-connected graphs
- **Balanced Connectivity**: Average 1.11 edges per node

### **Content Quality**
- **Role Coverage**: 100% of documents have all major roles
- **Relation Coverage**: 100% of documents have both structural and semantic relations
- **Section Structure**: 99.8% multi-section documents (realistic)

### **Processing Quality**
- **Success Rate**: 100% (3,048/3,048 files processed)
- **Error Rate**: 0% (no processing failures)
- **Consistency**: All files follow same schema and processing pipeline

---

## 📁 **Output Files for Research**

### **JSON Files**
- `comprehensive_paper_statistics.json` - Complete statistics
- `corrected_graph_analysis.json` - Basic analysis
- `final_pipeline_report.json` - Pipeline summary

### **CSV Files** (in `paper_tables/` directory)
- `dataset_summary.csv` - Dataset overview
- `graph_structure_summary.csv` - Graph metrics
- `relation_type_summary.csv` - Relation distribution
- `role_summary.csv` - Role distribution
- `section_distribution.csv` - Section analysis
- `confidence_summary.csv` - Confidence statistics
- `document_length_summary.csv` - Document analysis
- `role_pair_matrix_full.csv` - Complete role pair matrix
- `top_role_pairs.csv` - Top role pair combinations
- `role_transition_probabilities.csv` - Transition probabilities

### **LaTeX Tables**
- Ready-to-use LaTeX table code for research papers
- Formatted for direct inclusion in academic publications

---

## 🚀 **Research Applications**

### **Graph Neural Networks**
- **RCR-GAT**: Role-Conditioned Relational Graph Attention Network
- **CSRA**: Contextualized Semantic Retrieval Agent
- **Node Classification**: CTU role prediction
- **Link Prediction**: Relation type prediction

### **Natural Language Processing**
- **Relation Extraction**: Government scheme relationships
- **Document Understanding**: Multi-section document analysis
- **Semantic Analysis**: Role-based content understanding

### **Government Informatics**
- **Policy Analysis**: Scheme structure and relationships
- **Compliance Checking**: Prerequisite and eligibility flows
- **Benefit Analysis**: Assistance and support relationships

---

## 🔒 **Dataset Status**

- **Status**: PRODUCTION READY - FROZEN
- **Version**: v2.0 (Final)
- **Quality Assurance**: ✅ PASSED
- **Research Ready**: ✅ YES
- **Modifications**: ❌ NONE ALLOWED

---

**Generated by**: CTU FlowRAG Pipeline v2.0  
**Analysis Date**: October 2, 2025  
**Dataset Size**: 3,048 documents, 187,220 CTUs, 191,128 relations  
**Ready for**: RCR-GAT/CSRA training, Graph neural network research, Government informatics analysis
