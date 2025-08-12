import joblib
from exp_offset_model import ExpOffsetModel  # import from module, not __main__

a_opt = 0.00141461
b_opt = 0.152591
c_opt = 0

model = ExpOffsetModel(a_opt, b_opt, c_opt)
joblib.dump(model, "../models/exp_offset_model.pkl")
print("Model saved.")
