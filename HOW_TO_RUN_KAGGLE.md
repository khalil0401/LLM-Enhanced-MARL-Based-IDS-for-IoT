# How to Run the Kaggle Notebook - Step-by-Step Guide

## 🎯 Quick Start (5 Steps)

1. **Create Kaggle Account** (if needed)
2. **Upload Notebook to Kaggle**
3. **Enable GPU + Internet**
4. **Add Dataset** (optional for now)
5. **Run All Cells**

**Total Time:** ~10 minutes setup + 6-8 hours training

---

## 📋 Detailed Step-by-Step Instructions

### Step 1: Create Kaggle Account (2 minutes)

1. Go to https://www.kaggle.com
2. Click **"Register"** (top-right)
3. Sign up with:
   - Google account (recommended)
   - Email + password
   - OR GitHub account
4. Verify your email
5. Complete profile (optional)

✅ **Done!** You now have free access to GPU computing.

---

### Step 2: Upload Notebook to Kaggle (1 minute)

**Option A: Direct Upload (Recommended)**

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"** (top-right)
3. Click **"File" → "Upload Notebook"** (top menu)
4. Navigate to: `C:\Users\ATECH STORE\Desktop\Project\notebooks\kaggle_training_template.ipynb`
5. Click **"Upload"**
6. Rename if desired (e.g., "LLM-MARL-IDS-Training")

**Option B: Import from GitHub**

1. Click **"New Notebook"**
2. Click **"File" → "Import Notebook"**
3. Enter URL: `https://github.com/khalil0401/LLM-Enhanced-MARL-Based-IDS-for-IoT/blob/main/notebooks/kaggle_training_template.ipynb`
4. Click **"Import"**

✅ **Done!** Notebook is now in your Kaggle account.

---

### Step 3: Configure Notebook Settings (1 minute)

**CRITICAL: Must enable GPU and Internet**

1. Look at **right sidebar** in notebook
2. Under **"Session Options"** section:
   
   **a) Enable GPU:**
   - Click dropdown under "Accelerator"
   - Select **"GPU T4 x2"** or **"GPU P100"**
   - ✅ Confirm GPU is enabled
   
   **b) Enable Internet:**
   - Toggle **"Internet"** switch to **ON**
   - This is needed to download packages and clone GitHub repo
   - Internet can be disabled after Cell 2 completes
   
   **c) Set Persistence (Optional):**
   - Under "Persistence", select **"Variables only"** or **"Files only"**

3. Click **"Save Version"** to apply settings

✅ **Done!** Notebook is configured for training.

**Visual Guide:**
```
Right Sidebar:
┌─────────────────────────┐
│ Session Options         │
├─────────────────────────┤
│ Accelerator:            │
│ [GPU T4 x2  ▼]  ← SELECT│
├─────────────────────────┤
│ Internet:               │
│ [●ON  ○OFF]    ← TOGGLE │
├─────────────────────────┤
│ Persistence:            │
│ [Variables only ▼]      │
└─────────────────────────┘
```

---

### Step 4: Prepare Dataset (Optional - Can Skip for Now)

You have **two options**:

**Option A: Use Synthetic Data (Quick Start)**
- The code will auto-generate synthetic data
- No setup needed
- Good for testing the pipeline
- **Skip to Step 5!**

**Option B: Use Real IoT Dataset (For Research)**
- Need to download and upload IoT-23 dataset
- See detailed instructions in `KAGGLE_TRAINING_GUIDE.md`
- Takes 1-2 hours for dataset prep
- Better for actual experiments

**For first run, use Option A (synthetic data).**

---

### Step 5: Run the Notebook (6-8 hours)

**Method 1: Run All Cells at Once (Recommended)**

1. Click **"Run All"** button (top toolbar)
2. Or press **Shift + Ctrl + Enter** repeatedly
3. Monitor progress in output panels

**Method 2: Run Cell by Cell (For Learning)**

1. Click first cell
2. Press **Shift + Enter** to run and move to next
3. Wait for cell to complete (spinner stops)
4. Repeat for each cell

**Cell-by-Cell Breakdown:**

```
Cell 1: Install Dependencies          (~2-3 minutes)
  ├─ Installs PyTorch, Ray, Transformers
  ├─ Output: "✅ Dependencies installed!"
  └─ No errors expected

Cell 2: Clone GitHub Repository       (~30 seconds)
  ├─ Clones your project code
  ├─ Output: Repository structure listing
  └─ No errors expected

Cell 3: Configure for Kaggle          (~5 seconds)
  ├─ Sets up training configuration
  ├─ Output: Config summary
  └─ No errors expected

Cell 4: Check GPU and Resources       (~5 seconds)
  ├─ Verifies GPU is available
  ├─ Output: GPU name, memory, RAM
  └─ Expected: "✅ GPU Available!"

Cell 5: Train MAPPO Model             (~6-8 hours) ⏰
  ├─ Main training loop
  ├─ Output: Training progress, rewards
  ├─ Saves checkpoints every 10 iterations
  └─ Expected: "✅ Training complete!"

Cell 6: Load Phi-3-mini                (~1 minute)
  ├─ Downloads Phi-3-mini model (~7GB)
  ├─ Output: Model parameters, memory usage
  └─ Expected: "✅ Phi-3-mini loaded!"

Cell 7: Test Explanation               (~10 seconds)
  ├─ Generates sample explanation
  ├─ Output: Formatted alert with explanation
  └─ Expected: Generation time ~0.7 seconds

Cell 8: Evaluate with Explanations     (~30 seconds)
  ├─ Generates 10 test explanations
  ├─ Output: Average generation time, cost savings
  └─ Expected: "✅ Generated 10 explanations"

Cell 9: Save Results                   (~1 minute)
  ├─ Copies checkpoints to output
  ├─ Output: Files saved to /kaggle/working/results
  └─ Expected: "✅ Results saved"

Cell 10: Cleanup GPU Memory            (~5 seconds)
  ├─ Frees GPU memory
  ├─ Output: GPU memory allocation
  └─ Expected: Lower memory usage
```

---

### Step 6: Monitor Progress During Training

**What to Watch:**

1. **Session Timer (Top-Right Corner)**
   ```
   [11:23:45 remaining]  ← Don't let this hit 0:00:00!
   ```
   - Maximum: 12 hours per session
   - Save checkpoint before time runs out

2. **GPU Usage**
   - Check Cell 4 output periodically
   - Should show ~8-10GB GPU usage during training

3. **Training Progress (Cell 5)**
   ```
   Iteration 10: Reward=12.5, Episode Length=150
   Iteration 20: Reward=18.3, Episode Length=145
   ...
   ```
   - Reward should generally increase
   - Episode length may vary

4. **Checkpoint Saving**
   ```
   Checkpoint saved: /kaggle/working/checkpoints/checkpoint_000010
   Checkpoint saved: /kaggle/working/checkpoints/checkpoint_000020
   ...
   ```
   - Happens every 10 iterations
   - CRITICAL for resuming if session times out

**If Session Times Out:**
- Don't panic! Checkpoints are saved every 10 iterations
- See "Step 8: Resume Training" below

---

### Step 7: Download Results (5 minutes)

**After training completes:**

1. Look at **top-right corner** of Kaggle
2. Click **"Output"** tab
3. You'll see folder: `/kaggle/working/results/`
4. Click **"Download All"** button
5. Save to your computer

**What You'll Get:**
```
results/
├── checkpoints/
│   ├── checkpoint_000100/
│   ├── checkpoint_000200/
│   └── checkpoint_000500/  ← Final checkpoint
├── training_summary.json
└── (other logs and metrics)
```

**File Sizes:**
- Total: ~2-5 GB
- Each checkpoint: ~500 MB - 1 GB
- Logs: ~10-50 MB

---

### Step 8: Resume Training (If Session Timed Out)

**If your 12-hour session ends before training completes:**

1. **Download checkpoint** (see Step 7)
2. **Upload checkpoint as Kaggle Dataset:**
   - Go to https://www.kaggle.com/datasets
   - Click **"New Dataset"**
   - Upload checkpoint folder
   - Name: `mappo-checkpoint-iter500` (or your iteration number)
   - Make it **Private** or **Public**

3. **Create new notebook session:**
   - Open your notebook again
   - Click **"Edit"** to start new session

4. **Add checkpoint dataset:**
   - Click **"Add Data"** (right sidebar)
   - Search for your checkpoint dataset
   - Click **"+"** to add

5. **Modify Cell 3:**
   ```python
   # Add this line to config
   'resume_from_checkpoint': '/kaggle/input/mappo-checkpoint-iter500/checkpoint_000500'
   ```

6. **Run notebook again** - it will resume from checkpoint!

---

## 🎯 Quick Reference Commands

### Check GPU Status
```python
!nvidia-smi
```

### Check Remaining Time
```python
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

### Force Save Checkpoint
```python
# Add to training cell if needed
if iteration % 10 == 0:
    algo.save('/kaggle/working/checkpoints')
```

### Check Disk Space
```python
!df -h /kaggle/working
```

---

## ⚠️ Common Issues and Solutions

### Issue 1: "No GPU Available"
**Symptom:** Cell 4 shows "❌ No GPU!"

**Solution:**
1. Stop notebook (top bar: Session Options → Stop)
2. Enable GPU (right sidebar: Accelerator → GPU T4 x2)
3. Restart notebook
4. Run cells again

---

### Issue 2: "Dataset not found"
**Symptom:** Cell 4 shows "❌ Dataset not found!"

**Solution:**
- For first run: **Ignore this!** Code will use synthetic data
- For real dataset: See `KAGGLE_TRAINING_GUIDE.md` for upload instructions

---

### Issue 3: "Out of Memory" during Phi-3-mini loading
**Symptom:** Error in Cell 6

**Solution:**
```python
# In Cell 5 (before Phi-3-mini), add cleanup:
import torch
torch.cuda.empty_cache()
```

Or run Cell 10 (cleanup) before Cell 6.

---

### Issue 4: Session timeout before completion
**Symptom:** Session disconnects after 12 hours

**Solution:**
- This is normal!
- Download checkpoint (see Step 7)
- Resume training (see Step 8)
- OR reduce iterations: Change `total_iterations: 500` to `200` in Cell 3

---

### Issue 5: "Rate limit exceeded" for model download
**Symptom:** Phi-3-mini download fails

**Solution:**
1. Wait 1 hour
2. Try again
3. OR use different model:
   ```python
   model_name = "microsoft/phi-2"  # Smaller, 2.7B params
   ```

---

## 📊 Expected Outputs

### Successful Cell 4 Output:
```
✅ GPU Available!
   GPU Name: Tesla T4
   GPU Memory: 15.0 GB

✅ RAM: 13.0 GB
   Available: 11.5 GB

❌ Dataset not found! Add dataset in 'Add Data' section
   Expected path: /kaggle/input/iot23-processed/iot23_processed.h5
   
Note: Will use synthetic data for this run
```

### Successful Cell 5 Output:
```
============================================================
Starting MAPPO Training on Kaggle
============================================================

Iteration    0: Reward= 5.2, Episode Length=200
Iteration   10: Reward=12.5, Episode Length=180
Iteration   20: Reward=18.3, Episode Length=165
...
Iteration  500: Reward=45.7, Episode Length=120

✅ Training complete!
   Total time: 6.8 hours
   Final checkpoint: /kaggle/working/checkpoints/checkpoint_000500
```

### Successful Cell 7 Output:
```
============================================================
CRITICAL ALERT - IoT IDS (Local LLM)
============================================================
Device: 192.168.1.42 (IP Camera)
Severity: 100/100
Confidence: 0.87

--- Explanation (Phi-3-mini) ---
The IP camera shows abnormal DNS activity with 500 queries per second...

⚡ Generation time: 734ms
💰 Cost: $0 (GPT-4 would cost $0.01)
```

---

## 💡 Tips for Successful Training

### Before Starting:
1. ✅ **Verify GPU enabled** (right sidebar)
2. ✅ **Verify Internet enabled** (for Cells 1-2)
3. ✅ **Check session time** (aim for >10 hours remaining)
4. ✅ **Save notebook version** (File → Save Version)

### During Training:
1. 🔍 **Monitor session timer** every hour
2. 📊 **Check training progress** - rewards should increase
3. 💾 **Verify checkpoints saving** every 10 iterations
4. 🚨 **If <30 min remaining:** Download checkpoint immediately!

### After Training:
1. 💾 **Download all results** before closing session
2. 📝 **Note final iteration number**
3. 🔄 **Upload checkpoint if resuming later**
4. 🎉 **Celebrate - you saved $5,330!**

---

## 🎯 Summary Checklist

**Before Running:**
- [ ] Kaggle account created
- [ ] Notebook uploaded to Kaggle
- [ ] GPU enabled (GPU T4 x2 or P100)
- [ ] Internet enabled
- [ ] Settings saved

**During Run:**
- [ ] Cell 1: Dependencies installed (~3 min)
- [ ] Cell 2: Repo cloned (~30 sec)
- [ ] Cell 3: Config created (~5 sec)
- [ ] Cell 4: GPU verified (~5 sec)
- [ ] Cell 5: Training started (~6-8 hours)
- [ ] Cell 6: Phi-3-mini loaded (~1 min)
- [ ] Cell 7-8: Explanations generated (~1 min)
- [ ] Cell 9: Results saved (~1 min)

**After Completion:**
- [ ] Downloaded results from Output tab
- [ ] Saved checkpoint (if resuming later)
- [ ] Noted final metrics
- [ ] Checked training summary.json

---

## 🚀 You're Ready!

**Next Step:** Go to https://www.kaggle.com/code and upload the notebook!

**Estimated Total Time:**
- Setup: 10 minutes
- Training: 6-8 hours (automated)
- Download results: 5 minutes

**Total Cost:** **$0** (vs $5,330 traditional approach)

**Questions?** Check `KAGGLE_TRAINING_GUIDE.md` for detailed troubleshooting.

---

**Good luck with your training! 🎓🚀**
