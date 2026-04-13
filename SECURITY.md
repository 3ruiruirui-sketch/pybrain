# Security Policy

## Supported Versions

| Version | Supported          | Notes |
|---------|-------------------|-------|
| 0.1.x   | :white_check_mark: | Initial release |

---

## Responsible Disclosure

This is research software for brain tumor MRI analysis. While the software
is not certified as a medical device, we take security seriously.

If you discover a vulnerability that could affect patient safety or data
security, please **do not open a public GitHub issue**. Instead, contact the
maintainers privately.

### Reporting a Vulnerability

Please report security issues via email or private GitHub security advisories.
Include as much detail as possible:

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested fixes (optional)

We aim to acknowledge reports within 48 hours and provide a timeline for
resolution.

---

## Critical Disclaimers

### :warning: NOT A MEDICAL DEVICE

**PY-BRAIN IS RESEARCH SOFTWARE AND IS NOT:**

- **Cleared or approved by the FDA, EMA, or any regulatory body**
- **Certified as a medical device under MDR, 21 CFR Part 820, or equivalent**
- **Intended for clinical diagnosis, treatment decisions, or patient care**

### :warning: RESEARCH USE ONLY

All output from this pipeline must be interpreted by a qualified radiologist
or neuro-oncologist. AI-generated segmentations may contain errors,
particularly in:

- Highly infiltrative tumors
- Tumors near anatomical boundaries
- Cases with imaging artifacts or unusual contrast uptake patterns
- Post-operative or treatment-effect cases

### :warning: DATA SECURITY

- **Never include patient DICOM data in public repositories or bug reports**
- Ensure DICOM data is de-identified before processing
- Handle all imaging data in compliance with GDPR, HIPAA, and applicable
  data protection regulations
- Store results securely with appropriate access controls

---

## Known Security Considerations

### Model Weights

Model weights are downloaded from external sources (MONAI Cloud, BrainIAC
release page) on first run. Verify checksums against published values before
use in research.

### Third-Party Dependencies

The pipeline depends on several third-party packages including PyTorch, MONAI,
pydicom, and others. Keep dependencies updated to receive security patches.

### Input Validation

The pipeline assumes input NIfTI files conform to the BraTS format
specification. Malformed inputs may cause unexpected behavior.

---

## Compliance Notes

This software was developed as an experimental research tool. It has not
undergone formal validation required for clinical deployment. Any
institutional use should be reviewed by your institution's compliance,
legal, and clinical engineering teams.

For clinical or commercial use cases, significant additional validation,
quality management system implementation, and regulatory clearance would be
required.
