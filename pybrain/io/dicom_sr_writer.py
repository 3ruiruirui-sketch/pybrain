# pybrain/io/dicom_sr_writer.py
"""
DICOM Structured Report (SR) writer for exporting quantitative measurements
using TID 1500 (Measurement Report) template for PACS integration.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

from pybrain.io.logging_utils import get_logger


logger = get_logger("pybrain")


def _create_code_sequence(value: str, scheme: str, meaning: str) -> Dataset:
    """Create a DICOM Code Sequence."""
    seq = Dataset()
    seq.CodeValue = value
    seq.CodingSchemeDesignator = scheme
    seq.CodeMeaning = meaning
    return seq


def _create_measurement(
    name_code: Dataset,
    value: float,
    unit_code: Dataset,
) -> Dataset:
    """Create a DICOM Measurement item."""
    meas = Dataset()
    meas.ConceptNameCodeSequence = [name_code]
    meas.MeasurementUnitsCodeSequence = [unit_code]
    meas.NumericValue = str(value)
    return meas


def write_measurement_report(
    measurements: Dict[str, Any],
    source_dicom_dir: Path,
    segmentation_dicom_path: Path,
    output_path: Path,
    series_description: str = "PY-BRAIN Tumor Measurements (Research Only)",
    manufacturer: str = "PY-BRAIN",
    manufacturer_model_name: str = "py-brain v2",
    software_versions: str = "2.0.0",
    include_disclaimer: bool = True,
) -> Path:
    """
    Write quantitative measurements as a DICOM-SR file using TID 1500 template.
    
    Args:
        measurements: Dictionary of measurement values (volumes in cc, uncertainty, etc.)
        source_dicom_dir: Directory containing source DICOM series
        segmentation_dicom_path: Path to the DICOM-SEG segmentation file
        output_path: Output path for DICOM-SR file
        series_description: Series description for the DICOM-SR
        manufacturer: Manufacturer name
        manufacturer_model_name: Manufacturer model name
        software_versions: Software version
        include_disclaimer: Whether to include research-only disclaimer
    
    Returns:
        Path to the written DICOM-SR file
    """
    logger.info(f"Writing DICOM-SR to {output_path}")
    
    # Load source DICOM series
    source_files = sorted(source_dicom_dir.glob("*.dcm"))
    if not source_files:
        raise ValueError(f"No DICOM files found in {source_dicom_dir}")
    
    logger.info(f"Found {len(source_files)} DICOM files in source directory")
    
    # Load first DICOM to get study/series metadata
    first_ds = pydicom.dcmread(str(source_files[0]))
    
    # Load segmentation DICOM for reference
    seg_ds = pydicom.dcmread(str(segmentation_dicom_path))
    
    # Create DICOM-SR dataset
    sr_ds = Dataset()
    
    # SOP Common
    sr_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.33"  # DICOM SR Storage
    sr_ds.SOPInstanceUID = generate_uid()
    
    # Study and Series
    sr_ds.StudyInstanceUID = first_ds.StudyInstanceUID
    sr_ds.SeriesInstanceUID = generate_uid()
    sr_ds.SeriesNumber = 1
    sr_ds.SeriesDescription = series_description
    
    # Equipment
    sr_ds.Manufacturer = manufacturer
    sr_ds.ManufacturerModelName = manufacturer_model_name
    sr_ds.SoftwareVersions = software_versions
    
    # SR Document General
    sr_ds.Modality = "SR"
    sr_ds.InstanceNumber = 1
    
    # SR Document Content - TID 1500 Measurement Report
    sr_ds.ValueType = "CONTAINER"
    sr_ds.ConceptNameCodeSequence = [
        _create_code_sequence(
            value="126000",
            scheme="DCM",
            meaning="Measurement Report",
        )
    ]
    
    # Observation Context - add as first item in ContentSequence
    obs_context = Dataset()
    obs_context.RelationshipType = "CONTAINS"
    obs_context.ValueType = "CONTAINER"
    obs_context.ConceptNameCodeSequence = [
        _create_code_sequence(
            value="121038",
            scheme="DCM",
            meaning="Algorithm",
        )
    ]
    
    # Observer Type
    obs_context.ObserverType = "DEVICE"
    obs_context.DeviceObserverName = manufacturer_model_name
    
    # Algorithm Identification
    alg_id = Dataset()
    alg_id.RelationshipType = "CONTAINS"
    alg_id.ValueType = "CODE"
    alg_id.ConceptNameCodeSequence = [
        _create_code_sequence(
            value="Algorithm Identification",
            scheme="99PYBRAIN",
            meaning="Algorithm Identification",
        )
    ]
    alg_id.ConceptCodeSequence = [
        _create_code_sequence(
            value="PY-BRAIN v2",
            scheme="99PYBRAIN",
            meaning="PY-BRAIN v2 Brain Tumor Segmentation",
        )
    ]
    
    obs_context.ContentSequence = [alg_id]
    
    # Add observation context to content sequence
    content_seq = [obs_context]
    
    # Procedure Code
    sr_ds.ProcedureCodeSequence = [
        _create_code_sequence(
            value="Whole Tumor Segmentation",
            scheme="SCT",
            meaning="Whole Tumor Segmentation",
        )
    ]
    
    # Image Library - Reference source series
    sr_ds.CurrentRequestedProcedureEvidenceSequence = []
    for ds_path in source_files:
        ds = pydicom.dcmread(str(ds_path))
        ref = Dataset()
        ref.ReferencedSOPClassUID = ds.SOPClassUID
        ref.ReferencedSOPInstanceUID = ds.SOPInstanceUID
        sr_ds.CurrentRequestedProcedureEvidenceSequence.append(ref)
    
    # Reference the DICOM-SEG
    seg_ref = Dataset()
    seg_ref.ReferencedSOPClassUID = seg_ds.SOPClassUID
    seg_ref.ReferencedSOPInstanceUID = seg_ds.SOPInstanceUID
    sr_ds.CurrentRequestedProcedureEvidenceSequence.append(seg_ref)
    
    # Volume unit code (SCT 121214 "Volume")
    volume_unit = _create_code_sequence(
        value="121214",
        scheme="SCT",
        meaning="Volume",
    )
    
    # Measurement Group: Whole Tumor
    if "wt_volume_cc" in measurements:
        wt_group = Dataset()
        wt_group.RelationshipType = "CONTAINS"
        wt_group.ValueType = "CONTAINER"
        wt_group.ConceptNameCodeSequence = [
            _create_code_sequence(
                value="Whole Tumor",
                scheme="SCT",
                meaning="Whole Tumor",
            )
        ]
        
        # Measurement: WT volume
        wt_meas = _create_measurement(
            name_code=_create_code_sequence(
                value="Whole Tumor Volume",
                scheme="SCT",
                meaning="Whole Tumor Volume",
            ),
            value=measurements["wt_volume_cc"],
            unit_code=volume_unit,
        )
        wt_group.ContentSequence = [wt_meas]
        content_seq.append(wt_group)
    
    # Measurement Group: Tumor Core
    if "tc_volume_cc" in measurements:
        tc_group = Dataset()
        tc_group.RelationshipType = "CONTAINS"
        tc_group.ValueType = "CONTAINER"
        tc_group.ConceptNameCodeSequence = [
            _create_code_sequence(
                value="Tumor Core",
                scheme="SCT",
                meaning="Tumor Core",
            )
        ]
        
        # Measurement: TC volume
        tc_meas = _create_measurement(
            name_code=_create_code_sequence(
                value="Tumor Core Volume",
                scheme="SCT",
                meaning="Tumor Core Volume",
            ),
            value=measurements["tc_volume_cc"],
            unit_code=volume_unit,
        )
        tc_group.ContentSequence = [tc_meas]
        content_seq.append(tc_group)
    
    # Measurement Group: Enhancing Tumor
    if "et_volume_cc" in measurements:
        et_group = Dataset()
        et_group.RelationshipType = "CONTAINS"
        et_group.ValueType = "CONTAINER"
        et_group.ConceptNameCodeSequence = [
            _create_code_sequence(
                value="Enhancing Tumor",
                scheme="SCT",
                meaning="Enhancing Tumor",
            )
        ]
        
        # Measurement: ET volume
        et_meas = _create_measurement(
            name_code=_create_code_sequence(
                value="Enhancing Tumor Volume",
                scheme="SCT",
                meaning="Enhancing Tumor Volume",
            ),
            value=measurements["et_volume_cc"],
            unit_code=volume_unit,
        )
        et_group.ContentSequence = [et_meas]
        content_seq.append(et_group)
    
    # Measurement Group: Necrotic Core
    if "nc_volume_cc" in measurements:
        nc_group = Dataset()
        nc_group.RelationshipType = "CONTAINS"
        nc_group.ValueType = "CONTAINER"
        nc_group.ConceptNameCodeSequence = [
            _create_code_sequence(
                value="Necrotic Core",
                scheme="SCT",
                meaning="Necrotic Core",
            )
        ]
        
        # Measurement: NC volume
        nc_meas = _create_measurement(
            name_code=_create_code_sequence(
                value="Necrotic Core Volume",
                scheme="SCT",
                meaning="Necrotic Core Volume",
            ),
            value=measurements["nc_volume_cc"],
            unit_code=volume_unit,
        )
        nc_group.ContentSequence = [nc_meas]
        content_seq.append(nc_group)
    
    # Measurement Group: Uncertainty
    if "uncertainty_mean" in measurements:
        unc_group = Dataset()
        unc_group.RelationshipType = "CONTAINS"
        unc_group.ValueType = "CONTAINER"
        unc_group.ConceptNameCodeSequence = [
            _create_code_sequence(
                value="Uncertainty",
                scheme="SCT",
                meaning="Uncertainty",
            )
        ]
        
        # Measurement: Mean uncertainty (dimensionless)
        unc_meas = _create_measurement(
            name_code=_create_code_sequence(
                value="Mean Uncertainty",
                scheme="SCT",
                meaning="Mean Uncertainty",
            ),
            value=measurements["uncertainty_mean"],
            unit_code=_create_code_sequence(
                value="1",
                scheme="UCUM",
                meaning="dimensionless",
            ),
        )
        unc_group.ContentSequence = [unc_meas]
        content_seq.append(unc_group)
    
    sr_ds.ContentSequence = content_seq
    
    # Add research-only disclaimer if requested
    if include_disclaimer:
        if not "(Research Only)" in sr_ds.SeriesDescription:
            sr_ds.SeriesDescription += " (Research Only)"
    
    # Save DICOM-SR file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sr_ds.save_as(str(output_path), write_like_original=False, implicit_vr=True, little_endian=True)
    
    logger.info(f"DICOM-SR written successfully to {output_path}")
    logger.info(f"  Series Description: {sr_ds.SeriesDescription}")
    logger.info(f"  Number of measurement groups: {len(content_seq)}")
    
    return output_path
