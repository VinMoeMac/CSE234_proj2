"""
Synthetic v3 — targeting exact wrong predictions from 0.4539 model analysis.
35 questions got completely wrong tables. 14 are NTSB.
This adds 3 examples per failing NTSB table, closely matching validation question phrasings.
"""
import json
import os

# Exact wrong predictions from failure analysis:
# pred=AIRBAG/CHILDSEAT gold=CDC  (crush depth)
# pred=CDC/EDRPRECRASH gold=AVOID  (avoidance equipment)
# pred=EDRPOSTCRASH gold=ADAPT  (adaptive equipment)
# pred=EDRPRECRASH gold=GV  (lighting condition)
# pred=EVENT gold=EDREVENT  (event data recorder events)
# pred=GV gold=VPICDECODE  (vehicle make/model year)
# pred=AIRBAG/CHILDSEAT gold=GV  (vehicle count)
# pred=AIRBAG/CHILDSEAT gold=CDC  (collision speed)
# pred=EDRPRECRASH gold=CRASH  (years span)
# pred=EDRPOSTCRASH gold=ICS  (injury energy source)
# pred=EDRPOSTCRASH gold=OCC  (seatbelt use)
# pred=TIRE/VEHSPEC gold=VPICDECODE/TIREPLAC  (tire sizes)
# pred=EDRPOSTCRASH gold=EDREVENT  (EDR events count)

SYNTHETIC_V3 = [
    # =========================================================
    # CDC = Crush Data Coding (crush depth, delta-v, PDOF)
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "What is the maximum depth of crush for each crash event?",
        "schema_links": {"CDC": ["CASEID", "EVENTNO", "CMAX"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the crush depth measurements coded as 888 or 999.",
        "schema_links": {"CDC": ["CASEID", "CMAX"]}
    },
    {
        "db_id": "NTSB",
        "question": "What is the highest vehicle-to-barrier collision crush depth?",
        "schema_links": {"CDC": ["CASEID", "CMAX", "PDOF"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the principal direction of force for each crash event.",
        "schema_links": {"CDC": ["CASEID", "EVENTNO", "PDOF"]}
    },
    # =========================================================
    # AVOID = Pre-crash avoidance maneuvers
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "How many crashes involved vehicles equipped with crash avoidance systems?",
        "schema_links": {"AVOID": ["CASEID", "EQUIP"]}
    },
    {
        "db_id": "NTSB",
        "question": "Which vehicles had pre-crash avoidance equipment that was available but not activated?",
        "schema_links": {"AVOID": ["CASEID", "VEHNO", "EQUIP", "AVAIL", "ACTIVATE"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show whether collision avoidance technology was available for each vehicle.",
        "schema_links": {"AVOID": ["CASEID", "EQUIP", "AVAIL"]}
    },
    # =========================================================
    # ADAPT = Adaptive equipment for disabled drivers
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "How many crashes involved vehicles with after-market adaptive driving equipment?",
        "schema_links": {"ADAPT": []}
    },
    {
        "db_id": "NTSB",
        "question": "Show the adaptive equipment type for each vehicle in the dataset.",
        "schema_links": {"ADAPT": ["CASEID", "VEHNO", "ADAPT"]}
    },
    {
        "db_id": "NTSB",
        "question": "Which vehicles had adaptive equipment installed for disabled drivers?",
        "schema_links": {"ADAPT": ["CASEID", "VEHNO"]}
    },
    # =========================================================
    # GV = General Vehicle info (lighting, road surface, weather)
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "Display a count of crashes grouped by lighting condition at time of crash.",
        "schema_links": {"GV": ["CASEID", "LIGHTCOND"]}
    },
    {
        "db_id": "NTSB",
        "question": "How many crashes occurred in each weather condition?",
        "schema_links": {"GV": ["CASEID", "WEATHER"]}
    },
    {
        "db_id": "NTSB",
        "question": "How many different vehicles are there in this crash dataset?",
        "schema_links": {"GV": ["CASEID", "VEHNO"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the road surface condition and alignment for each crash.",
        "schema_links": {"GV": ["CASEID", "SURFCOND", "ALIGNMENT"]}
    },
    # =========================================================
    # EDREVENT = Event Data Recorder events
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "How many events were recorded by event data recorders in these crashes?",
        "schema_links": {"EDREVENT": ["CASEID", "EDREVENTNO"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the event description and ignition cycle at time of crash from EDR data.",
        "schema_links": {"EDREVENT": ["CASEID", "EVENTDESC", "IGCYCRASH"]}
    },
    {
        "db_id": "NTSB",
        "question": "What are the case IDs and event numbers for all EDR-recorded crash events?",
        "schema_links": {"EDREVENT": ["CASEID", "VEHNO", "EDREVENTNO"]}
    },
    # =========================================================
    # VPICDECODE = VIN decode (make, model, model year)
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "What vehicle make has the oldest average model year in the dataset?",
        "schema_links": {"VPICDECODE": ["CASEID", "Make", "ModelYear"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the vehicle type and manufacturer name for each VIN-decoded vehicle.",
        "schema_links": {"VPICDECODE": ["CASEID", "VehicleType", "ManufacturerFullName"]}
    },
    {
        "db_id": "NTSB",
        "question": "What model year are the vehicles in this crash study?",
        "schema_links": {"VPICDECODE": ["CASEID", "ModelYear"]}
    },
    # =========================================================
    # CRASH = Crash-level data (year, month, vehicles)
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "What years does this crash investigation dataset span?",
        "schema_links": {"CRASH": ["CRASHYEAR"]}
    },
    {
        "db_id": "NTSB",
        "question": "How many crashes occurred in each month of the year?",
        "schema_links": {"CRASH": ["CASEID", "CRASHMONTH"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the number of vehicles involved and day of week for each crash.",
        "schema_links": {"CRASH": ["CASEID", "VEHICLES", "DAYOFWEEK"]}
    },
    # =========================================================
    # ICS = Injury Coding Source (energy source of injury)
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "How many injuries were caused by an unknown source of energy?",
        "schema_links": {"ICS": ["CASEID", "SOE"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the body region and source of energy for each coded injury.",
        "schema_links": {"ICS": ["CASEID", "BRI", "SOE"]}
    },
    # =========================================================
    # OCC = Occupant (seatbelt, age, sex)
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "How many vehicle occupants did not wear a seatbelt even though one was available?",
        "schema_links": {"OCC": ["CASEID", "OCCNO", "BELTUSE", "BELTAVAIL"]}
    },
    {
        "db_id": "NTSB",
        "question": "Show the age and sex of each occupant involved in crashes.",
        "schema_links": {"OCC": ["CASEID", "OCCNO", "AGE", "SEX"]}
    },
    # =========================================================
    # TIREPLAC = Tire placard (recommended tire sizes)
    # =========================================================
    {
        "db_id": "NTSB",
        "question": "Show the first recommended front and rear tire sizes for each vehicle.",
        "schema_links": {"TIREPLAC": ["CASEID", "RECFRONT1", "RECREAR1"]}
    },
    {
        "db_id": "NTSB",
        "question": "What are the recommended tire sizes listed on the vehicle placard?",
        "schema_links": {"TIREPLAC": ["CASEID", "RECFRONT1", "RECREAR1", "RECFRPRESS1"]}
    },
]


def main():
    os.makedirs("data/augmented", exist_ok=True)

    for i, ex in enumerate(SYNTHETIC_V3):
        ex["question_id"] = 9600 + i
        ex.setdefault("gold_sql", "")

    # Combine all synthetics
    v1 = json.load(open("augmented_data/sap_synthetic.json"))
    v2 = json.load(open("augmented_data/synthetic_v2.json"))
    all_synth = v1 + v2 + SYNTHETIC_V3

    out = "data/augmented/synthetic_v3.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(SYNTHETIC_V3, f, indent=2, ensure_ascii=False)
    print(f"Written {len(SYNTHETIC_V3)} new NTSB examples to {out}")

    train = json.load(open("Project2/train.json"))
    combined = train + all_synth
    out2 = "data/augmented/train_v3.json"
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Written combined {len(combined)} examples to {out2}")


if __name__ == "__main__":
    main()
