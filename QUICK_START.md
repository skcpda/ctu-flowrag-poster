# 🚀 Quick Start Guide - Role Labeling System

## **1. Set Your API Key**
```bash
export OPENAI_API_KEY='your-actual-api-key-here'
```

## **2. Install Dependencies**
```bash
pip install -r requirements_labeling.txt
```

## **3. Run the System**
```bash
python run_role_labeling.py
```

## **4. Choose Your Option**
- **Option 1**: Test with 5 sample files (~$0.50)
- **Option 2**: Process all 2,038 schemes (~$2.32)
- **Option 3**: Analyze existing results

---

## **📊 Cost Breakdown**
- **GPT-3.5-Turbo**: $2.32 total for all 23,761 sentences
- **GPT-4o**: $8.61 total (use only for complex cases)

## **🎯 What You Get**
- **23 role categories** (ProblemContext, Objective, Benefit, etc.)
- **Structured slots** (amounts, rates, criteria, steps)
- **Role probabilities** for confidence scoring
- **Number detection** and currency normalization

---

**Ready to process your government schemes!** 🎉
