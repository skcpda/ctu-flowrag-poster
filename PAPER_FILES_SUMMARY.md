# 📊 Research Paper Files Summary

**Generated**: October 2, 2025  
**Purpose**: Complete statistics and data for research paper

---

## 📁 **Generated Files Overview**

### **Main Statistics Files**
1. **`comprehensive_paper_statistics.json`** - Complete detailed statistics
2. **`corrected_graph_analysis.json`** - Basic graph analysis
3. **`final_pipeline_report.json`** - Pipeline processing summary

### **Research Documentation**
4. **`RESEARCH_PAPER_STATISTICS.md`** - Comprehensive statistics report
5. **`PRODUCTION_READY_GRAPH_SUMMARY.md`** - Dataset overview and status
6. **`PAPER_FILES_SUMMARY.md`** - This file listing all generated files

### **CSV Data Files** (in `paper_tables/` directory)
7. **`dataset_summary.csv`** - Dataset overview metrics
8. **`graph_structure_summary.csv`** - Graph density and structure metrics
9. **`relation_type_summary.csv`** - Relation type distribution with percentages
10. **`role_summary.csv`** - Role distribution with percentages
11. **`section_distribution.csv`** - Section count distribution across files
12. **`confidence_summary.csv`** - Confidence statistics (raw vs calibrated)
13. **`document_length_summary.csv`** - Document length statistics
14. **`role_pair_matrix_full.csv`** - Complete role pair co-occurrence matrix
15. **`top_role_pairs.csv`** - Top role pair combinations by frequency
16. **`role_transition_probabilities.csv`** - Role transition probability matrix

---

## 📊 **Key Statistics for Paper**

### **Dataset Scale**
- **3,048 government scheme documents**
- **187,220 Content Thematic Units (CTUs)**
- **191,128 relations** (89.2% structural, 10.8% semantic)
- **Average 61.4 CTUs per document**
- **Average 5.5 sections per document**

### **Graph Structure**
- **Average density**: 0.017139 (sparse, controlled)
- **Average semantic ratio**: 10.8%
- **99.9% multi-section documents** (realistic structure)
- **Section-bounded PRECEDES edges** (no cross-section jumps)

### **Relation Types**
- **PRECEDES**: 170,483 (89.2%) - Sequential flow
- **PREREQUISITE_OF**: 11,656 (6.1%) - Prerequisite relationships
- **ADMINISTERED_BY**: 5,638 (3.0%) - Administrative authority
- **CAP_LIMITS**: 1,688 (0.9%) - Financial/benefit limits
- **ENABLES**: 1,663 (0.9%) - Enabling relationships

### **Role Distribution**
- **ContextObjective**: 75,892 (40.5%) - Scheme context
- **BenefitsAssistance**: 48,526 (25.9%) - Benefits details
- **Eligibility**: 21,564 (11.5%) - Eligibility criteria
- **AuthoritiesGovernance**: 16,887 (9.0%) - Administrative authorities
- **ApplicationProcess**: 15,848 (8.5%) - Application procedures
- **TimelineFrequency**: 7,473 (4.0%) - Timeline information
- **DefinitionsReferences**: 1,030 (0.6%) - Technical definitions

### **Confidence Analysis**
- **Raw confidence**: 0.934 ± 0.050 (0.70-0.95)
- **Calibrated confidence**: 0.748 ± 0.040 (0.56-0.76)
- **Structural edges**: High confidence (0.7-1.0)
- **Semantic edges**: Medium confidence (0.3-0.7)

---

## 🎯 **Ready-to-Use LaTeX Tables**

The analysis also generated LaTeX table code that can be directly included in your research paper:

### **Table 1: Dataset Overview**
```latex
\begin{table}[h]
\centering
\caption{Dataset Overview}
\begin{tabular}{|l|r|}
\hline
Metric & Value \\
\hline
Total Documents & 3,048 \\
Total CTUs & 187,220 \\
Total Relations & 191,128 \\
Structural Relations & 170,483 \\
Semantic Relations & 20,645 \\
Avg CTUs per Document & 61.4 \\
Avg Relations per Document & 62.7 \\
Avg Sections per Document & 5.5 \\
\hline
\end{tabular}
\end{table}
```

### **Table 2: Relation Type Distribution**
```latex
\begin{table}[h]
\centering
\caption{Relation Type Distribution}
\begin{tabular}{|l|r|r|}
\hline
Relation Type & Count & Percentage \\
\hline
PRECEDES & 170,483 & 89.2\% \\
PREREQUISITE_OF & 11,656 & 6.1\% \\
ADMINISTERED_BY & 5,638 & 3.0\% \\
CAP_LIMITS & 1,688 & 0.9\% \\
ENABLES & 1,663 & 0.9\% \\
ELABORATES & 0 & 0.0\% \\
SUPPORTS & 0 & 0.0\% \\
\hline
\end{tabular}
\end{table}
```

### **Table 3: Role Distribution**
```latex
\begin{table}[h]
\centering
\caption{Role Distribution}
\begin{tabular}{|l|r|r|}
\hline
Role & Count & Percentage \\
\hline
ContextObjective & 75,892 & 40.5\% \\
BenefitsAssistance & 48,526 & 25.9\% \\
Eligibility & 21,564 & 11.5\% \\
AuthoritiesGovernance & 16,887 & 9.0\% \\
ApplicationProcess & 15,848 & 8.5\% \\
TimelineFrequency & 7,473 & 4.0\% \\
DefinitionsReferences & 1,030 & 0.6\% \\
\hline
\end{tabular}
\end{table}
```

---

## 📈 **Research Applications**

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

## 📝 **Usage Instructions**

1. **For LaTeX papers**: Use the generated LaTeX table code directly
2. **For data analysis**: Use the CSV files in `paper_tables/` directory
3. **For comprehensive stats**: Reference `comprehensive_paper_statistics.json`
4. **For methodology**: Reference `RESEARCH_PAPER_STATISTICS.md`

---

**All files are ready for your research paper!** 🎉
