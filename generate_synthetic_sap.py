"""
Generate synthetic NL question / schema_links pairs for SAP databases.
These are hand-authored based on column semantics — no external API needed.
Output: data/augmented/sap_synthetic.json (same format as train.json)
"""
import json
import os

SAP_SYNTHETIC = [
    # SBODemoUS-Human Resources
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "What are the first name, last name, and salary of all employees?",
        "schema_links": {"OHEM": ["firstName", "lastName", "salary"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "List the email address and mobile number of each employee.",
        "schema_links": {"OHEM": ["email", "mobile", "empID"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Which employees have a termination date set?",
        "schema_links": {"OHEM": ["empID", "firstName", "lastName", "termDate"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Show the department and branch for each employee.",
        "schema_links": {"OHEM": ["empID", "dept", "branch"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "What is the job title and start date of each employee?",
        "schema_links": {"OHEM": ["empID", "jobTitle", "startDate"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "List all employee types and their descriptions.",
        "schema_links": {"OHTY": ["typeID", "name", "descriptio"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Show all team names and their descriptions.",
        "schema_links": {"OHTM": ["teamID", "name", "descriptio"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Which employees are members of which teams?",
        "schema_links": {"HTM1": ["teamID", "empID", "role"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Show the education history including institute and diploma for each employee.",
        "schema_links": {"HEM2": ["empID", "institute", "diploma", "major"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "What positions are available in the company?",
        "schema_links": {"OHPS": ["posID", "name", "descriptio"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Show employees with their home city and home phone number.",
        "schema_links": {"OHEM": ["empID", "homeCity", "homeTel"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "List the termination reasons available.",
        "schema_links": {"OHTR": ["reasonID", "name", "descriptio"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "What is the salary and salary unit for employees in each department?",
        "schema_links": {"OHEM": ["dept", "salary", "salaryUnit", "empID"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Show the work city and work state of all active employees.",
        "schema_links": {"OHEM": ["empID", "workCity", "workState", "status"]}
    },
    # SBODemoUS-Reports
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show all query names and their query strings.",
        "schema_links": {"OUQR": ["QName", "QString", "QType"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "List the query categories and how many queries are in each.",
        "schema_links": {"OUQR": ["QCategory"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show the action code and parameters for each report action.",
        "schema_links": {"SRA1": ["ActionCode", "ParamCode", "KeyString"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "What are the report names and their authors?",
        "schema_links": {"SRT2": ["DocName", "Author"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show documents that have email conversion enabled.",
        "schema_links": {"SRT2": ["DocCode", "DocName", "SwpInEmail"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "List report documents with their extension error action and number of repetitive areas.",
        "schema_links": {"SRT2": ["DocCode", "ExtOnErr", "NumRepArs"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show all query authorization groups and their codes.",
        "schema_links": {"OQAG": ["AUTHGRPID", "AUTHGRPCD", "AUTHGRPN"]}
    },
    # SBODemoUS-Service
    {
        "db_id": "SBODemoUS-Service",
        "question": "List all service contract IDs and their start and end dates.",
        "schema_links": {"OCTR": ["ContractID", "StartDate", "EndDate"]}
    },
    {
        "db_id": "SBODemoUS-Service",
        "question": "Show the customer code and status for each service call.",
        "schema_links": {"OSCL": ["callID", "CustCode", "Status"]}
    },
    {
        "db_id": "SBODemoUS-Service",
        "question": "How many service contract items are there per contract?",
        "schema_links": {"CTR1": ["ContractID"]}
    },
    {
        "db_id": "SBODemoUS-Service",
        "question": "Show the queue name and manager for each service queue.",
        "schema_links": {"OQUT": ["QueueID", "QueueName", "Manager"]}
    },
    # SBODemoUS-Finance
    {
        "db_id": "SBODemoUS-Finance",
        "question": "Show all journal entry transaction numbers and their posting dates.",
        "schema_links": {"OJDT": ["TransId", "RefDate"]}
    },
    {
        "db_id": "SBODemoUS-Finance",
        "question": "List the account codes and their balances.",
        "schema_links": {"OACT": ["AcctCode", "CurrTotal"]}
    },
    {
        "db_id": "SBODemoUS-Finance",
        "question": "Show recurring postings with their frequency and amount.",
        "schema_links": {"OFRC": ["Frequency", "Amount", "TransCode"]}
    },
    # SBODemoUS-General
    {
        "db_id": "SBODemoUS-General",
        "question": "List all business partners with their names and group codes.",
        "schema_links": {"OCRD": ["CardCode", "CardName", "GroupCode"]}
    },
    {
        "db_id": "SBODemoUS-General",
        "question": "Show the item codes, names, and item groups.",
        "schema_links": {"OITM": ["ItemCode", "ItemName", "ItmsGrpCod"]}
    },
    # SBODemoUS-Sales Opportunities
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "Show the open date, close date, and closing percentage for all opportunities.",
        "schema_links": {"OPR1": ["OpenDate", "CloseDate", "ClosPrcnt"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "What is the weighted sum in local currency for each sales opportunity?",
        "schema_links": {"OPR1": ["WtSumLoc", "OpprId"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "Show the status and gross profit for each opportunity.",
        "schema_links": {"OPR1": ["Status", "GrosProfit", "OpprId"]}
    },
    # SBODemoUS-Banking
    {
        "db_id": "SBODemoUS-Banking",
        "question": "List all incoming payments with their amounts and dates.",
        "schema_links": {"ORCT": ["DocNum", "DocDate", "CashSum"]}
    },
    {
        "db_id": "SBODemoUS-Banking",
        "question": "Show bank account numbers and their current balances.",
        "schema_links": {"OACT": ["AcctCode", "CurrTotal", "AcctName"]}
    },
    # SBODemoUS-Inventory and Production
    {
        "db_id": "SBODemoUS-Inventory and Production",
        "question": "List all warehouses with their names and locations.",
        "schema_links": {"OWHS": ["WhsCode", "WhsName", "Street"]}
    },
    {
        "db_id": "SBODemoUS-Inventory and Production",
        "question": "Show item codes, names, and their current stock quantity.",
        "schema_links": {"OITM": ["ItemCode", "ItemName", "OnHand"]}
    },
    # SBODemoUS-Business Partners
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "Show all sales employees and their names.",
        "schema_links": {"OSLP": ["SlpCode", "SlpName"]}
    },
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "List customer codes with their credit limits and balance.",
        "schema_links": {"OCRD": ["CardCode", "CreditLine", "Balance"]}
    },
]


def main():
    os.makedirs("data/augmented", exist_ok=True)

    # assign question IDs starting after train.json max
    for i, ex in enumerate(SAP_SYNTHETIC):
        ex["question_id"] = 9000 + i
        ex.setdefault("gold_sql", "")

    out_path = "data/augmented/sap_synthetic.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(SAP_SYNTHETIC, f, indent=2, ensure_ascii=False)

    print(f"Written {len(SAP_SYNTHETIC)} synthetic SAP examples to {out_path}")

    # also write a combined train+synthetic file
    with open("Project2/train.json", "r", encoding="utf-8") as f:
        train = json.load(f)

    combined = train + SAP_SYNTHETIC
    combined_path = "data/augmented/train_with_sap.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Written combined {len(combined)} examples to {combined_path}")


if __name__ == "__main__":
    main()
