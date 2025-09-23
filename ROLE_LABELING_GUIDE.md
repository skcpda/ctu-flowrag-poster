# Role Labeling System for Government Schemes

## 🎯 **OVERVIEW**

This system labels sentences from government welfare schemes with primary roles and extracts structured slots using OpenAI's GPT models.

## 📊 **PRICING ANALYSIS**

### **Cost Estimates for 23,761 Sentences:**

| Model | Input Cost | Output Cost | **Total Cost** |
|-------|------------|-------------|-----------------|
| **GPT-3.5-Turbo** | $0.89 | $1.43 | **$2.32** |
| **GPT-4o** | $1.49 | $7.13 | **$8.61** |

### **Recommendation:**
- **Start with GPT-3.5-turbo** for cost efficiency (~$2.32 total)
- **Use GPT-4o** only for complex/ambiguous cases
- **Batch size 15-25** sentences for optimal balance

---

## 🚀 **QUICK START**

### **1. Setup Environment**
```bash
# Install dependencies
pip install -r requirements_labeling.txt

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

### **2. Run the System**
```bash
# Interactive menu
python run_role_labeling.py

# Or process sample files directly
python -c "from run_role_labeling import process_sample_files; process_sample_files()"
```

---

## 🔧 **USAGE OPTIONS**

### **Option 1: Sample Processing (Recommended for Testing)**
- Processes 5 files for testing
- Cost: ~$0.50
- Time: ~5 minutes

### **Option 2: Full Processing**
- Processes all 2,038 scheme files
- Cost: ~$2.32 (GPT-3.5-turbo)
- Time: ~2-3 hours

### **Option 3: Analysis**
- Analyzes existing labeled results
- Shows role distribution and statistics

---

## 📋 **ROLE LABELING SYSTEM**

### **Allowed Roles (23 total):**
- **ProblemContext** - Describes the problem being addressed
- **Objective** - Goals and aims of the scheme
- **Benefit** - Financial benefits, amounts, rates
- **Eligibility** - Who can apply, criteria, requirements
- **ApplicationProcess** - How to apply, steps, channels
- **Timeline** - Deadlines, cycles, duration
- **ContactsGovernance** - Authorities, committees, contact info
- **Exclusion** - Who is not eligible
- **Definition** - Terms and definitions
- **ImplementingAgencyJurisdiction** - Agencies and jurisdictions
- **FinancialDetails** - Cost breakdowns, GST info
- **RequiredDocuments** - Documents needed
- **VerificationInspection** - Verification processes
- **DisbursalComputation** - Payment calculations
- **ComplianceConditions** - Compliance requirements
- **AppealsGrievance** - Appeal processes
- **TargetBeneficiariesSector** - Target groups and sectors
- **GeographyScope** - Geographic coverage
- **GovernanceBodies** - Governance structures
- **Mode** - Application modes
- **FrequencyCycle** - Frequency and cycles
- **FootnoteLegalBasis** - Legal references
- **Misc** - Everything else

### **Slot Extraction Examples:**

#### **Benefit Slots:**
```json
{
  "amount": 625000.0,
  "rate_percent": 12.5,
  "cap_amount": 1000000.0,
  "periodicity": "monthly",
  "included_costs": ["tuition", "books", "transport"]
}
```

#### **Eligibility Slots:**
```json
{
  "subject": "students",
  "age": 18,
  "income_cap": 500000.0,
  "unit_type": "annual",
  "registration": "Aadhaar required",
  "geo_scope": "India",
  "exceptions": ["government employees"]
}
```

#### **ApplicationProcess Slots:**
```json
{
  "steps": ["Register online", "Upload documents", "Submit application"],
  "channel": "online",
  "office": "District Education Office",
  "form": "Form A",
  "fee": 0.0
}
```

---

## 📁 **OUTPUT STRUCTURE**

### **Input Format:**
```json
{
  "doc_id": "scheme-name",
  "sentences": [
    {"sid": "S1", "text": "Scheme title"},
    {"sid": "S2", "text": "Scheme description..."}
  ]
}
```

### **Output Format:**
```json
{
  "doc_id": "scheme-name",
  "labels": [
    {
      "sid": "S1",
      "role": "Objective",
      "role_probs": {
        "Objective": 0.8,
        "Benefit": 0.15,
        "Eligibility": 0.05
      },
      "slots": {
        "goal": "Provide financial assistance",
        "target": "students"
      },
      "has_numbers": false
    }
  ]
}
```

---

## 🎯 **BEST PRACTICES**

### **Cost Optimization:**
1. **Start with GPT-3.5-turbo** for initial processing
2. **Use batch size 15-25** for optimal balance
3. **Process in phases** to monitor quality
4. **Cache similar patterns** to avoid reprocessing

### **Quality Assurance:**
1. **Review sample results** before full processing
2. **Monitor role distribution** for consistency
3. **Validate slot extraction** for accuracy
4. **Use GPT-4o** for complex cases only

### **Batch Processing:**
- **Small batches (5-10)**: Better accuracy, higher cost
- **Medium batches (15-25)**: Balanced cost/accuracy
- **Large batches (30-50)**: Lower cost, potential accuracy loss

---

## 📈 **MONITORING & ANALYSIS**

### **Role Distribution Analysis:**
```bash
python run_role_labeling.py
# Choose option 3: Analyze existing results
```

### **Expected Results:**
- **Top roles**: Eligibility, Benefit, ApplicationProcess
- **Distribution**: Should be balanced across policy areas
- **Quality**: High confidence scores for primary roles

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues:**

1. **API Key Not Found**
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

2. **Rate Limiting**
   - Increase delay between batches
   - Reduce batch size
   - Use smaller batches

3. **JSON Parsing Errors**
   - Check API response format
   - Verify model compatibility
   - Review prompt structure

4. **High Costs**
   - Switch to GPT-3.5-turbo
   - Increase batch size
   - Process in smaller chunks

---

## 📊 **EXPECTED OUTCOMES**

### **Processing Results:**
- **23,761 sentences** labeled with roles
- **Structured slots** extracted for each role
- **Role probabilities** for confidence scoring
- **Number detection** for financial data

### **Quality Metrics:**
- **High accuracy** for clear policy statements
- **Consistent role assignment** across similar content
- **Comprehensive slot extraction** for structured data
- **Proper normalization** of currency and numbers

---

## 🎯 **NEXT STEPS**

1. **Run sample processing** to test the system
2. **Review results** for quality and accuracy
3. **Adjust parameters** if needed
4. **Run full processing** on all schemes
5. **Analyze results** for insights and patterns

---

**Ready to start role labeling your government schemes!** 🚀
