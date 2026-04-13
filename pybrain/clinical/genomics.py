class BrainIAC_Predictor:
    @staticmethod
    def predict_idh(features, patient_age):
        is_mismatch = features.get("t2_flair_mismatch", False)
        ct_calc_vol_cc = features.get("calcification_present", False)
        
        idh_prob = 0.85 if (is_mismatch or ct_calc_vol_cc) else 0.12
        
        if patient_age > 75:
            idh_prob *= 0.20
            
        if idh_prob > 0.50:
            status = "Mutante (IDH-Mutant)"
        else:
            status = "Selvagem (Wildtype)"
            
        return {"status": status, "confidence": idh_prob if idh_prob > 0.50 else 1.0 - idh_prob}
