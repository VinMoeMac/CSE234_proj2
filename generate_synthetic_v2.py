"""
Generate synthetic v2 — targets NTSB gaps and remaining SAP 0.000 databases.
Focus on failure patterns identified from 0.440 model analysis.
"""
import json
import os

SYNTHETIC_V2 = [
    # =========================================================
    # NTSB — targeting the 13 wrong-table failures
    # GV = General Vehicle info (make, model, year, lighting)
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "How many different vehicles are there in total?",
        "schema_links": {"GV": ["CASEID", "VEHNO"]}
    },
    {
        "db_id": "NTSB",
        "question": "Display a count of crashes by lighting condition.",
        "schema_links": {"GV": ["CASEID", "LIGHTCOND"]}
    },
    {
        "db_id": "NTSB",
        "question": "What vehicle makes and models are in the dataset?",
        "schema_links": {"GV": ["VPICMAKE", "VPICMODEL"]}
    },
    {
        "db_id": "NTSB",
        "question": "What is the average model year of vehicles involved in crashes?",
        "schema_links": {"GV": ["CASEID", "VEHNO"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the vehicle make and body class for each vehicle.",
        "schema_links": {"GV": ["VPICMAKE", "VPICBODYCLASS", "VEHNO"]}
    },
    {
        "db_id": "NTSB",
        "question": "What road surface conditions were present during crashes?",
        "schema_links": {"GV": ["CASEID", "SURFCOND"]}
    },
    {
        "db_id": "NTSB",
        "question": "What is the speed limit at crash locations?",
        "schema_links": {"GV": ["CASEID", "SPEEDLIMIT"]}
    },
    # ADAPT = Adaptive equipment (for disabled drivers)
    {
        "db_id": "NTSB",
        "question": "How many crashes involved vehicles with after-market adaptive equipment?",
        "schema_links": {"ADAPT": []}
    },
    {
        "db_id": "NTSB",
        "question": "Show the adaptive equipment type for each vehicle.",
        "schema_links": {"ADAPT": ["CASEID", "VEHNO", "ADAPT"]}
    },
    {
        "db_id": "NTSB",
        "question": "Which vehicles had adaptive equipment installed?",
        "schema_links": {"ADAPT": ["CASEID", "VEHNO"]}
    },
    # AVOID = Avoidance maneuver / collision avoidance equipment
    {
        "db_id": "NTSB",
        "question": "How many crashes involved vehicles equipped with collision avoidance systems?",
        "schema_links": {"AVOID": ["CASEID", "EQUIP"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show whether avoidance equipment was available and activated.",
        "schema_links": {"AVOID": ["CASEID", "EQUIP", "AVAIL", "ACTIVATE"]}
    },
    {
        "db_id": "NTSB",
        "question": "Which vehicles had pre-crash avoidance maneuvers?",
        "schema_links": {"AVOID": ["CASEID", "VEHNO", "AVAIL"]}
    },
    # EDREVENT = Event Data Recorder event data
    {
        "db_id": "NTSB",
        "question": "How many events were recorded by event data recorders?",
        "schema_links": {"EDREVENT": ["CASEID", "EDREVENTNO"]}
    },
    {
        "db_id": "NTSB",
        "question": "What are the case IDs and event descriptions from event data recorders?",
        "schema_links": {"EDREVENT": ["CASEID", "EVENTDESC", "EDREVENTNO"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the ignition cycle and warning lamp status at time of crash.",
        "schema_links": {"EDREVENT": ["CASEID", "IGCYCRASH", "WARNLAMP"]}
    },
    # ICS = Injury Coding Source
    {
        "db_id": "NTSB",
        "question": "How many injuries were caused by an unknown energy source?",
        "schema_links": {"ICS": ["CASEID", "SOE"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the body region index and source of energy for each injury.",
        "schema_links": {"ICS": ["CASEID", "BRI", "SOE"]}
    },
    {
        "db_id": "NTSB",
        "question": "What is the injury coding confidence level for each occupant?",
        "schema_links": {"ICS": ["CASEID", "OCCNO", "ICSCONFIDENCE"]}
    },
    # OCC = Occupant data
    {
        "db_id": "NTSB",
        "question": "How many vehicle occupants did not wear a seatbelt?",
        "schema_links": {"OCC": ["CASEID", "OCCNO", "BELTUSE"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the age, sex, and seat location of each occupant.",
        "schema_links": {"OCC": ["CASEID", "OCCNO", "AGE", "SEX", "SEATLOC"]}
    },
    {
        "db_id": "NTSB",
        "question": "What is the average age of vehicle occupants involved in crashes?",
        "schema_links": {"OCC": ["CASEID", "AGE"]}
    },
    # VPICDECODE = VIN decode / vehicle identification
    {
        "db_id": "NTSB",
        "question": "What vehicle make has the oldest average model year?",
        "schema_links": {"VPICDECODE": ["CASEID", "ManufacturerFullNameId", "ModelYear"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the vehicle type and manufacturer for each vehicle.",
        "schema_links": {"VPICDECODE": ["CASEID", "VehicleType", "ManufacturerFullNameId"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the first recommended front and rear tire sizes for each vehicle.",
        "schema_links": {"TIREPLAC": ["CASEID", "RECFRONT1", "RECREAR1"]}
    },
    # CDC = Crash Data Coding (crush/damage)
    {
        "db_id": "NTSB",
        "question": "What is the maximum depth of crush measured in crashes?",
        "schema_links": {"CDC": ["CASEID", "CMAX"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the principal direction of force for each crash event.",
        "schema_links": {"CDC": ["CASEID", "PDOF", "EVENTNO"]}
    },
    {
        "db_id": "NTSB",
        "question": "What is the highest vehicle-to-object collision speed delta-v?",
        "schema_links": {"CDC": ["CASEID", "ENDSHIFT"]}
    },
    # CRASH = crash-level data
    {
        "db_id": "NTSB",
        "question": "What years does this crash dataset span?",
        "schema_links": {"CRASH": ["CRASHYEAR"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the number of vehicles and day of week for each crash.",
        "schema_links": {"CRASH": ["CASEID", "VEHICLES", "DAYOFWEEK"]}
    },
    {
        "db_id": "NTSB",
        "question": "How many crashes occurred on each day of the week?",
        "schema_links": {"CRASH": ["CASEID", "DAYOFWEEK"]}
    },

    # =========================================================
    # SBODemoUS-Banking (still 0.000)
    # =========================================================
    {
        "db_id": "SBODemoUS-Banking",
        "question": "Show all checks with their account number and check date.",
        "schema_links": {"OCHO": ["CheckKey", "AcctNum", "CheckDate"]}
    },
    {
        "db_id": "SBODemoUS-Banking",
        "question": "List incoming payments with their document number and cash amount.",
        "schema_links": {"ORCT": ["DocNum", "DocDate", "CashSum"]}
    },
    {
        "db_id": "SBODemoUS-Banking",
        "question": "Show outgoing payments with their vendor code and total amount.",
        "schema_links": {"OVPM": ["DocNum", "CardCode", "DocTotal"]}
    },
    {
        "db_id": "SBODemoUS-Banking",
        "question": "What are the account numbers that have checks with more than one payment?",
        "schema_links": {"OCHO": ["AcctNum", "CheckKey"], "CHO1": ["CheckKey"]}
    },
    {
        "db_id": "SBODemoUS-Banking",
        "question": "Show payment run entries with their vendor number and payment amount.",
        "schema_links": {"OPEX": ["AbsEntry", "VendorNum", "PymDocAmnt"]}
    },

    # =========================================================
    # SBODemoUS-Sales Opportunities (still 0.000)
    # =========================================================
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "Show open and close dates for sales opportunities with closing percentage below 50.",
        "schema_links": {"OPR1": ["OpprId", "OpenDate", "CloseDate", "ClosePrcnt"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "What is the weighted amount in local currency for row-level opportunity data?",
        "schema_links": {"OPR1": ["OpprId", "WtSumLoc"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "Show all opportunity stages with their description.",
        "schema_links": {"OOST": ["Num", "Descript"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "List sales opportunities with their sales employee and predicted close date.",
        "schema_links": {"OPR1": ["OpprId", "SlpCode", "CloseDate"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "How many queries are there in each category?",
        "schema_links": {"OPR1": ["OpprId"]}
    },

    # =========================================================
    # SBODemoUS-Business Partners (still 0.000)
    # =========================================================
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "Show all sales employees with their name and code.",
        "schema_links": {"OSLP": ["SlpCode", "SlpName"]}
    },
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "List customers with their credit limit and current balance.",
        "schema_links": {"OCRD": ["CardCode", "CardName", "CreditLine", "Balance"]}
    },
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "Show the phone number and email for each business partner.",
        "schema_links": {"OCRD": ["CardCode", "Phone1", "E_Mail"]}
    },
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "What are the target amounts for each sales employee?",
        "schema_links": {"OSLP": ["SlpCode", "SlpName"]}
    },
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "Show the fax number and contact person for each business partner.",
        "schema_links": {"OCRD": ["CardCode", "Fax", "CntctPrsn"]}
    },
]


def main():
    os.makedirs("data/augmented", exist_ok=True)

    for i, ex in enumerate(SYNTHETIC_V2):
        ex["question_id"] = 9500 + i
        ex.setdefault("gold_sql", "")

    # Load existing SAP synthetic and combine
    existing = json.load(open("augmented_data/sap_synthetic.json"))
    combined_synth = existing + SYNTHETIC_V2

    out_synth = "data/augmented/synthetic_v2.json"
    with open(out_synth, "w", encoding="utf-8") as f:
        json.dump(SYNTHETIC_V2, f, indent=2, ensure_ascii=False)
    print(f"Written {len(SYNTHETIC_V2)} new examples to {out_synth}")

    # Full combined training set
    train = json.load(open("Project2/train.json"))
    combined = train + combined_synth
    out_combined = "data/augmented/train_v2.json"
    with open(out_combined, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Written combined {len(combined)} examples to {out_combined}")


if __name__ == "__main__":
    main()
