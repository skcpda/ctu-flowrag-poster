#!/bin/bash
"""
Comprehensive evaluation commands for CTU-FlowRAG system.
This script provides all the commands needed to run the full evaluation pipeline.
"""

set -e  # Exit on any error

echo "🚀 CTU-FlowRAG Comprehensive Evaluation Pipeline"
echo "================================================"

# Configuration
CONFIG_PATH="ctu_flowrag/configs/rcr_gat.yaml"
TENSOR_DIR="tensors_dev"
WEIGHTS_PATH="ckpts/rcr_gat_trained.pt"
OUTPUT_DIR="evaluation_results"
TEMPLATES="[[0,1,3],[0,2,3],[0,1,2],[0,4,5]]"

echo "📊 Configuration:"
echo "  Config: $CONFIG_PATH"
echo "  Tensor dir: $TENSOR_DIR"
echo "  Weights: $WEIGHTS_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Templates: $TEMPLATES"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/paper_artifacts
mkdir -p $OUTPUT_DIR/deliverables

echo "📊 Running comprehensive evaluation..."

# 1. Main evaluation (our model vs baselines)
echo "1️⃣ Running main evaluation..."
python -c "
import sys
sys.path.append('.')
from run_comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator('$CONFIG_PATH', '$TENSOR_DIR', '$WEIGHTS_PATH')
evaluator._run_main_evaluation()
evaluator._run_baseline_evaluation()
print('✅ Main evaluation completed')
"

# 2. Ablation studies
echo "2️⃣ Running ablation studies..."
python -c "
import sys
sys.path.append('.')
from run_comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator('$CONFIG_PATH', '$TENSOR_DIR', '$WEIGHTS_PATH')
evaluator._run_ablation_evaluation()
print('✅ Ablation studies completed')
"

# 3. Role coverage and faithfulness
echo "3️⃣ Running role coverage evaluation..."
python -c "
import sys
sys.path.append('.')
from run_comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator('$CONFIG_PATH', '$TENSOR_DIR', '$WEIGHTS_PATH')
evaluator._run_role_coverage_evaluation()
print('✅ Role coverage evaluation completed')
"

# 4. Per-role recall
echo "4️⃣ Running per-role recall evaluation..."
python -c "
import sys
sys.path.append('.')
from run_comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator('$CONFIG_PATH', '$TENSOR_DIR', '$WEIGHTS_PATH')
evaluator._run_per_role_recall_evaluation()
print('✅ Per-role recall evaluation completed')
"

# 5. Attention sanity
echo "5️⃣ Running attention sanity check..."
python -c "
import sys
sys.path.append('.')
from run_comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator('$CONFIG_PATH', '$TENSOR_DIR', '$WEIGHTS_PATH')
evaluator._run_attention_sanity_evaluation()
print('✅ Attention sanity check completed')
"

# 6. Robustness testing
echo "6️⃣ Running robustness testing..."
python -c "
import sys
sys.path.append('.')
from run_comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator('$CONFIG_PATH', '$TENSOR_DIR', '$WEIGHTS_PATH')
evaluator._run_robustness_evaluation()
print('✅ Robustness testing completed')
"

# 7. Generate paper artifacts
echo "7️⃣ Generating paper artifacts..."
python -c "
import sys
sys.path.append('.')
from run_comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator('$CONFIG_PATH', '$TENSOR_DIR', '$WEIGHTS_PATH')
evaluator._generate_paper_artifacts()
print('✅ Paper artifacts generated')
"

# 8. Generate deliverables
echo "8️⃣ Generating deliverables..."
python -c "
import sys
sys.path.append('.')
from run_comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator('$CONFIG_PATH', '$TENSOR_DIR', '$WEIGHTS_PATH')
evaluator._generate_deliverables()
print('✅ Deliverables generated')
"

# 9. Validate results
echo "9️⃣ Validating results against targets..."
python -c "
import sys
sys.path.append('.')
from run_comprehensive_evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator('$CONFIG_PATH', '$TENSOR_DIR', '$WEIGHTS_PATH')
validation = evaluator.validate_results_against_targets()

if validation['passed']:
    print('✅ All targets met! Model is ready for paper.')
else:
    print('⚠️ Some targets not met. Check validation_results.json')
    print('Issues:', validation['issues'])
"

echo ""
echo "📊 Evaluation Summary:"
echo "====================="
echo "📁 Results directory: $OUTPUT_DIR"
echo "📋 Main results: $OUTPUT_DIR/main_results.csv"
echo "📋 Baseline comparison: $OUTPUT_DIR/baseline_comparison.csv"
echo "📋 Ablation results: $OUTPUT_DIR/ablation_results.csv"
echo "📋 Role coverage: $OUTPUT_DIR/role_coverage_results.csv"
echo "📋 Per-role recall: $OUTPUT_DIR/per_role_recall.csv"
echo "📋 Attention sanity: $OUTPUT_DIR/attention_sanity.json"
echo "📋 Robustness: $OUTPUT_DIR/robustness_results.json"
echo "📋 Validation: $OUTPUT_DIR/validation_results.json"
echo ""
echo "📊 Paper Artifacts:"
echo "==================="
echo "📋 Table 1: $OUTPUT_DIR/paper_artifacts/table1_main_results.csv"
echo "📋 Table 2: $OUTPUT_DIR/paper_artifacts/table2_role_coverage.csv"
echo "📋 Table 3: $OUTPUT_DIR/paper_artifacts/table3_ablations.csv"
echo "📋 LaTeX sources: $OUTPUT_DIR/paper_artifacts/*.tex"
echo ""
echo "📊 Deliverables:"
echo "==============="
echo "📋 Per-doc results: $OUTPUT_DIR/deliverables/per_doc_results.csv"
echo "📋 Role coverage report: $OUTPUT_DIR/deliverables/role_coverage_report.json"
echo "📋 Per-role recall: $OUTPUT_DIR/deliverables/per_role_recall.csv"
echo "📋 Attention sanity: $OUTPUT_DIR/deliverables/attention_sanity_dump.json"
echo "📋 Robustness sweeps: $OUTPUT_DIR/deliverables/robustness_sweeps.json"
echo "📋 Efficiency report: $OUTPUT_DIR/deliverables/efficiency_report.json"
echo "📋 Case studies: $OUTPUT_DIR/deliverables/case_studies/"
echo ""
echo "🎉 Comprehensive evaluation completed!"
echo "📊 Check $OUTPUT_DIR/validation_results.json for target validation"
