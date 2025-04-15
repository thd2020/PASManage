import mysql.connector
import pandas as pd
from datetime import datetime
import os

mri_base_dir = "/home/lmj/xyx/sda2/placenta_mri_100"
mri_categories = ["穿透", "粘连", "正常", "植入"]

# Connect to MySQL database
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="thd2020",
    password="Pasq124p610",
    database="pas_manage_db"
)
cursor = conn.cursor()

# Load the Excel file
file_path = '胎盘植入16-24_V5.xlsx'
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name='Sheet1')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.reset_index(drop=True, inplace=True)

# Function to handle the enums (mapping for Yes/No conditions to 1/0)
def map_enum(field, value):
    if field == "referral_status":
        mapping = {"APPROVED": "APPROVED", "PENDING": "PENDING", "REJECTED": "REJECTED"}
    elif field == "placenta_previa":
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("1"):
                return 1  # "COMPLETE"
            elif value.startswith("2"):
                return 2  # "PARTIAL"
            elif value.startswith("3"):
                return 3  # "MARGINAL"
            elif value.startswith("4"):
                return 4  # "DANGEROUS"
        return None
    # Handling numeric or other data types
    else:
        return value  # Default case, return as is

# Function to handle blank/missing values
def handle_blank(value):
    return None if pd.isna(value) or value == '' else value

# Insert data into the tables
for _, row in df.iterrows():
    # Insert into 'patient' table
    patient_data = {
        'name': handle_blank(row['姓名']),
        'patient_id': handle_blank(row['ID']),
        'gender': 'FEMALE',
    }
    cursor.execute("""
        INSERT INTO patient (name, patient_id, gender)
        VALUES (%s, %s, %s)
    """, tuple(patient_data.values()))

    # Complete and correctly ordered dictionary for medical_record table
    medical_record_data = {
        # 'afp': handle_blank(row['AFP']),
        'age': handle_blank(row['年龄']),
        # 'anemia': map_enum('anemia', row.get('贫血')),
        'artery_embolization_history': handle_blank(row['介入后并发症']),
        # 'assisted_reproduction': handle_blank(row['辅助生殖']),  # if exists
        # 'b_hcg': handle_blank(row['bHCG']),  # if exists
        'c_section_count': handle_blank(row['剖宫产*']),
        'current_pregnancy_count': handle_blank(row['妊娠次数']),  # if exists
        'diabetes': handle_blank(row['妊娠期糖尿病  （有/无）']),
        'diagnosis': handle_blank(row['MRI报告*']),
        # 'fibroid_surgery_history': handle_blank(row['子宫肌瘤手术史']),  # if exists
        'gravidity': handle_blank(row['妊娠次数']),
        'height_cm': handle_blank(row['身高']),
        'hypertension': handle_blank(row['妊娠期高血压疾病  （有/无）']),
        # 'inhibina': handle_blank(row['抑制素A']),  # if exists
        'medical_abortion': handle_blank(row['药物流产']),
        'name': handle_blank(row['姓名']),
        'notes': None,
        # 'pas_history': handle_blank(row['PAS病史']),  # if exists
        'pre_delivery_weight': handle_blank(row['分娩前体重']),
        'surgical_abortion': handle_blank(row['中晚孕引产']),
        'symptoms': None,
        'treatment': None,
        # 'ue3': handle_blank(row['UE3']),  # if exists
        # 'uterine_surgery_history': handle_blank(row['子宫手术史']),  # if exists
        'vaginal_delivery_count': handle_blank(row['阴道分娩次数']),
        'visit_date': datetime.now(),  # default to now
        'visit_type': None,
        # 'doctor_id': handle_blank(row['主刀医师']),
        'patient_id': handle_blank(row['ID']),
        'placenta_previa': map_enum('placenta_previa', row['前置胎盘*（1.完全2.部分3.边缘4.凶险性前置胎盘）']),
    }
    cursor.execute(f"""
        INSERT INTO medical_record ({', '.join(medical_record_data.keys())})
        VALUES ({', '.join(['%s'] * len(medical_record_data))})
    """, tuple(medical_record_data.values()))

    # Insert into 'surgery_and_blood_test' table
    surgery_data = {
        'patient_id': handle_blank(row['ID']),
        'primary_surgeon': handle_blank(row['主刀医师']),
        'assisting_surgeon': None,  # not in Excel
        'gestational_weeks': handle_blank(row['分娩孕周']),
        'hospital_stay_days': handle_blank(row.get('住院天数')),  # optional, if exists

        'surgery_duration': handle_blank(row['手术时间（小时）']),
        'intraoperative_bleeding': handle_blank(row['术中出血量']),
        'pre_delivery_bleeding': handle_blank(row['产前出血']),

        'anesthesia_method': handle_blank(row.get('分娩麻醉方式')),
        'newborn_weight': handle_blank(row.get('新生儿体重g')),
        'arterial_ph': handle_blank(row.get('新生儿动脉血气PH')),
        'apgar_score': handle_blank(row.get('Apgar评分')),

        'preoperative_hb': handle_blank(row.get('术前HB')),
        'preoperative_hct': handle_blank(row.get('术前HCT')),
        'postoperative24h_hb': handle_blank(row.get('术后24h内Hb')),
        'postoperative24h_hct': handle_blank(row.get('术后24h内HCT')),
        'postoperative_transfusion_status': handle_blank(row['术后输血']),

        'cervical_surgery': handle_blank(row['宫颈提拉术']),
        'bilateral_uterine_artery_ligation': handle_blank(row['双侧子宫动脉结扎']),
        'bilateral_ovarian_artery_ligation': handle_blank(row['双侧卵巢子宫动脉交通支结扎']),
        'uterine_surgery_type': handle_blank(row['子宫前后壁排式']),
        'placenta_removal': handle_blank(row['手取胎盘术']),
        'uterine_reconstruction': handle_blank(row['子宫整形术']),
        'tubal_ligation': handle_blank(row['输卵管结扎']),
        'cook_balloon_sealing': handle_blank(row['cook球囊填塞']),
        'aortic_balloon': handle_blank(row['腹主动脉球囊']),
        'hysterectomy': handle_blank(row['子宫切除']),

        'red_blood_cells_transfused': handle_blank(row['输注红悬（u）']),
        'plasma_transfused': handle_blank(row['输注血浆（ml）']),
        'c_section_count': map_enum("c_section_count", handle_blank(row['剖宫次数'])),

        'notes': None  # or row.get('介入并发症后的费用（取栓后总手术费用）') if needed
    }

    cursor.execute(f"""
        INSERT INTO surgery_and_blood_test ({', '.join(surgery_data.keys())})
        VALUES ({', '.join(['%s'] * len(surgery_data))})
    """, tuple(surgery_data.values()))

    # Insert into 'ultrasound_score' table
    ultrasound_data = {
        'patient_id': handle_blank(row['ID']),
        'fetal_position': map_enum("fetal_position", handle_blank(row.get('是否头位'))),
        'placental_position': map_enum("placental_position", handle_blank(row.get('胎盘（中央/部分/边缘）'))),
        'total_score': handle_blank(row.get('总评分/预估出血量')),
        'examination_date': datetime.now(),

        'blood_flow_signals': handle_blank(row.get('胎盘与子宫肌壁间血流信号')),
        'cervical_shape': handle_blank(row.get('宫颈形态')),
        'estimated_blood_loss': None,  # this is often combined into total_score; keep as None unless separate column exists
        'myometrium_invasion': handle_blank(row.get('子宫肌层受侵犯程度')),
        'placental_body_position': map_enum("placental_body_position", handle_blank(row.get('胎盘主体前/后壁'))),
        'placental_lacunae': handle_blank(row.get('胎盘内陷窝、沸水征')),
        'placental_thickness': handle_blank(row.get('胎盘厚度')),
        'record_id': None,  # Optional: if you want to tie this to some MRI/record file, set this
        'suspected_invasion_range': handle_blank(row.get('可疑植入范围')),
        'suspected_placenta_location': handle_blank(row.get('可疑胎盘植入位置')),
    }

    cursor.execute(f"""
        INSERT INTO ultrasound_score ({', '.join(ultrasound_data.keys())})
        VALUES ({', '.join(['%s'] * len(ultrasound_data))})
    """, tuple(ultrasound_data.values()))

    patient_id = handle_blank(row['ID'])
    patient_name = handle_blank(row['姓名'])
    found_path = None
    result_description = None

    def is_valid_image_dir(path):
        return os.path.isdir(path) and len([f for f in os.listdir(path) if f.lower().endswith('.png')]) > 5

    # Try to locate the scan folder
    for category in mri_categories:
        category_path = os.path.join(mri_base_dir, category)
        if not os.path.exists(category_path):
            continue

        for folder in os.listdir(category_path):
            if patient_name in folder:
                patient_root = os.path.join(category_path, folder)
                for root, _, _ in os.walk(patient_root):
                    if is_valid_image_dir(root):
                        found_path = root
                        result_description = category
                        break
                if found_path:
                    break
        if found_path:
            break

    # If a path is found, insert imaging_record
    if found_path:
        image_count = len([f for f in os.listdir(found_path) if f.lower().endswith('.png')])
        imaging_data = {
            'record_id': f"{patient_id}_MRI",
            'image_count': image_count,
            'result_description': result_description,
            'test_date': datetime.now(),
            'test_type': 'MRI',
            'patient_id': patient_id,
            'path': found_path
        }

        cursor.execute(f"""
            INSERT INTO imaging_record ({', '.join(imaging_data.keys())})
            VALUES ({', '.join(['%s'] * len(imaging_data))})
        """, tuple(imaging_data.values()))

# Commit and close connection
conn.commit()
cursor.close()
conn.close()

print("Data inserted successfully!")
