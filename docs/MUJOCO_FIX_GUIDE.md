# MuJoCo Version Conflict Fix Guide

## Problem Summary

Your environment has **MuJoCo 3.3.7 installed** but is **loading MuJoCo 2.0 libraries** due to environment variables in your shell configuration. This causes the `apirate` attribute error because MuJoCo 2.x doesn't support modern XML schema features.

## Root Cause

Lines in `~/.zshrc`:
```bash
export DYLD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
```

These were likely added for old `mujoco-py` (MuJoCo 2.x) and are now interfering with the modern `mujoco` package (3.x).

---

## Fix Options

### Option 1: Quick Fix (Temporary - Current Session Only)

Run these commands in your **current terminal**:

```bash
# Unset the problematic variables
unset LD_LIBRARY_PATH
unset DYLD_LIBRARY_PATH

# Reinstall MuJoCo to ensure clean binary
pip uninstall -y mujoco
pip install mujoco>=3.1.0

# Test it works
python test_external_envs_integration.py --env shadow_hand_block
```

**Note**: This only fixes the current terminal session. Opening a new terminal will reload the old settings from `~/.zshrc`.

---

### Option 2: Permanent Fix (Recommended)

#### Step 1: Edit your shell configuration

Open `~/.zshrc` in an editor:

```bash
nano ~/.zshrc
# or
code ~/.zshrc
```

#### Step 2: Comment out or remove the MuJoCo 2.x lines

Find these lines:
```bash
export DYLD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
```

Either **comment them out** (add `#` at the beginning):
```bash
# export DYLD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$DYLD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
```

Or **delete them** entirely.

#### Step 3: Reload your shell configuration

```bash
source ~/.zshrc
```

Or simply **close and reopen your terminal**.

#### Step 4: Verify the fix

```bash
# Check that the variables are gone
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
# Should be empty or not contain mujoco200

# Check MuJoCo works
python check_mujoco_version.py
python diagnose_xml_issue.py
```

#### Step 5: Test the environment

```bash
python test_external_envs_integration.py --env shadow_hand_block
```

---

### Option 3: Hybrid Approach (Conditional Loading)

If you need **both** old MuJoCo 2.x (for legacy code) and new MuJoCo 3.x, you can use conditional loading:

Edit `~/.zshrc` and replace the lines with:

```bash
# Only load MuJoCo 2.x paths when specifically needed
# Uncomment the next two lines ONLY when running old mujoco-py code:
# export DYLD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$DYLD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
```

Or create a helper function:

```bash
# Add to ~/.zshrc
enable_mujoco2() {
    export DYLD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$DYLD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/Users/duruoli/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
    echo "MuJoCo 2.x paths enabled"
}

disable_mujoco2() {
    unset DYLD_LIBRARY_PATH
    unset LD_LIBRARY_PATH
    echo "MuJoCo 2.x paths disabled"
}
```

Then run `enable_mujoco2` or `disable_mujoco2` as needed.

---

## Additional Notes

### Why is libmujoco.3.3.7.dylib only 8.5 MB?

The small size (8.5 MB instead of 20-50 MB) suggests either:
1. It's a stub library that loads the actual binary from elsewhere
2. The installation is incomplete
3. It's a minimal build

The environment variables are causing it to load the old 2.0 binary instead of this one.

### Can I delete ~/.mujoco/mujoco200?

**Yes**, if you no longer need MuJoCo 2.x for old code. However, keep a backup first:

```bash
mv ~/.mujoco ~/.mujoco_backup
```

You can always restore it later if needed.

---

## Verification Commands

After applying the fix, verify everything works:

```bash
# 1. Check environment
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"

# 2. Check Python can import MuJoCo
python -c "import mujoco; print(f'Version: {mujoco.mj_versionString()}')"

# 3. Test apirate parsing
python -c "
import mujoco
xml = '''<mujoco model=\"test\">
  <option timestep=\"0.002\" apirate=\"200\"/>
  <worldbody>
    <body name=\"dummy\" pos=\"0 0 1\">
      <geom type=\"sphere\" size=\"0.1\"/>
    </body>
  </worldbody>
</mujoco>'''
try:
    model = mujoco.MjModel.from_xml_string(xml)
    print('✓ SUCCESS: apirate works!')
except Exception as e:
    print(f'✗ FAILED: {e}')
"

# 4. Test shadow hand environment
python test_external_envs_integration.py --env shadow_hand_block
```

---

## Recommended: Permanent Fix Steps

```bash
# 1. Edit ~/.zshrc and comment out the MuJoCo 2.x lines
code ~/.zshrc  # or nano ~/.zshrc

# 2. Reload shell
source ~/.zshrc

# 3. Verify variables are gone
env | grep -i mujoco

# 4. Reinstall MuJoCo (optional but recommended)
pip uninstall -y mujoco
pip install mujoco>=3.1.0

# 5. Test
python test_external_envs_integration.py --env shadow_hand_block
```

---

## Need Help?

If you still have issues after applying the fix:

1. Run the diagnostic scripts:
   - `python check_mujoco_version.py`
   - `python diagnose_xml_issue.py`
   - `python check_conflicts.py`

2. Check if there are other places setting these variables:
   ```bash
   grep -r "DYLD_LIBRARY_PATH\|LD_LIBRARY_PATH" ~/anaconda3/envs/hdm2/etc/conda/
   ```

3. Verify no other conda activation scripts are setting them.


