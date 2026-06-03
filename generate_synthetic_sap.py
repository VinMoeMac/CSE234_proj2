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
        "question": "What are the report document names and their authors?",
        "schema_links": {"RDOC": ["DocName", "Author"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show report documents that require email font conversion.",
        "schema_links": {"RDOC": ["DocCode", "DocName", "SwpInEmail", "EmailFont"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "List report documents with their extension error action and number of repetitive areas.",
        "schema_links": {"RDOC": ["DocCode", "DocName", "ExtOnErr", "NumRepArs"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show all query authorization groups and their codes.",
        "schema_links": {"OQAG": ["AUTHGRPID", "AUTHGRPCD", "AUTHGRPN"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "List all queries with their category and query string.",
        "schema_links": {"OUQR": ["QName", "QCategory", "QString"]}
    },
    # SBODemoUS-Service
    {
        "db_id": "SBODemoUS-Service",
        "question": "List all service contract IDs with customer name and status.",
        "schema_links": {"OCTR": ["ContractID", "CstmrName", "Status"]}
    },
    {
        "db_id": "SBODemoUS-Service",
        "question": "Show the customer name and subject for each service call.",
        "schema_links": {"OSCL": ["callID", "custmrName", "subject"]}
    },
    {
        "db_id": "SBODemoUS-Service",
        "question": "How many service contract items are there per contract?",
        "schema_links": {"CTR1": ["ContractID"]}
    },
    {
        "db_id": "SBODemoUS-Service",
        "question": "Show the queue description and manager for each service queue.",
        "schema_links": {"OQUE": ["queueID", "descript", "manager"]}
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
        "question": "Show recurring transaction templates with their frequency and last posted date.",
        "schema_links": {"ORCR": ["RcurCode", "RcurDesc", "Frequency", "LastPosted"]}
    },
    {
        "db_id": "SBODemoUS-Finance",
        "question": "List all projects with their project code and name.",
        "schema_links": {"OPRJ": ["PrjCode", "PrjName"]}
    },
    # SBODemoUS-Sales Opportunities
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "Show the open date, close date, and closing percentage for all opportunities.",
        "schema_links": {"OPR1": ["OpenDate", "CloseDate", "ClosePrcnt"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "What is the maximum sum in local currency for each sales opportunity?",
        "schema_links": {"OPR1": ["MaxSumLoc", "OpprId"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "Show the status and sales employee for each opportunity.",
        "schema_links": {"OPR1": ["OpprId", "SlpCode"]}
    },
    # SBODemoUS-Banking
    {
        "db_id": "SBODemoUS-Banking",
        "question": "List all incoming payments with their amounts and dates.",
        "schema_links": {"ORCT": ["DocNum", "DocDate", "CashSum"]}
    },
    {
        "db_id": "SBODemoUS-Banking",
        "question": "Show checks with their account number and check date.",
        "schema_links": {"OCHO": ["CheckKey", "AcctNum", "CheckDate"]}
    },
    # SBODemoUS-Inventory and Production
    {
        "db_id": "SBODemoUS-Inventory and Production",
        "question": "List all warehouses with their names.",
        "schema_links": {"OWHS": ["WhsCode", "WhsName"]}
    },
    {
        "db_id": "SBODemoUS-Inventory and Production",
        "question": "Show item codes and names with their current stock quantity.",
        "schema_links": {"OITM": ["ItemCode", "ItemName", "OnHand"]}
    },
    {
        "db_id": "SBODemoUS-Inventory and Production",
        "question": "Show production orders with their planned quantity and status.",
        "schema_links": {"OWOR": ["DocNum", "ItemCode", "PlannedQty", "Status"]}
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
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "What are the phone numbers and email addresses of business partners?",
        "schema_links": {"OCRD": ["CardCode", "CardName", "Phone1", "E_Mail"]}
    },
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "Show the sales employee code for each business partner.",
        "schema_links": {"OCRD": ["CardCode", "SlpCode"]}
    },
    # SBODemoUS-Human Resources (more examples for still-failing module)
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "List each employee's role ID and the date they were assigned to that role.",
        "schema_links": {"HEM6": ["empID", "roleID"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "What is the marital status and number of children for each employee?",
        "schema_links": {"OHEM": ["empID", "martStatus", "nChildren"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Show all employees and their citizenship and passport numbers.",
        "schema_links": {"OHEM": ["empID", "citizenshp", "passportNo"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "List employee performance reviews with grade and manager.",
        "schema_links": {"HEM3": ["empID", "grade", "manager", "reviewDesc"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Show all employee absence records with reason and approval.",
        "schema_links": {"HEM1": ["empID", "reason", "approvedBy", "fromDate", "toDate"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Which employees have a salary greater than 5000?",
        "schema_links": {"OHEM": ["empID", "firstName", "lastName", "salary"]}
    },
    {
        "db_id": "SBODemoUS-Human Resources",
        "question": "Show the previous employment history including employer and position for each employee.",
        "schema_links": {"HEM4": ["empID", "employer", "position", "fromDate", "toDate"]}
    },
    # SBODemoUS-General (still failing — use tables that actually exist)
    {
        "db_id": "SBODemoUS-General",
        "question": "List all business objects with their table name and primary key.",
        "schema_links": {"OBOB": ["ObjectId", "TableName", "PrimaryKey"]}
    },
    # SBODemoUS-Reports (still failing)
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show all queries with their category and last update date.",
        "schema_links": {"OUQR": ["QName", "QCategory", "QLastDate"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show the document code, author, and category for each report.",
        "schema_links": {"RDOC": ["DocCode", "Author", "Category"]}
    },
    # SBODemoUS-Sales Opportunities (still failing)
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "Show all opportunity stages with their description and step ID.",
        "schema_links": {"OOST": ["Num", "Descript", "StepId"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "What is the maximum sum in system currency for each sales opportunity?",
        "schema_links": {"OPR1": ["OpprId", "MaxSumSys"]}
    },
    # SBODemoUS-Inventory and Production (still failing)
    {
        "db_id": "SBODemoUS-Inventory and Production",
        "question": "List items with their unit of measure and last purchase price.",
        "schema_links": {"OITM": ["ItemCode", "ItemName", "SalUnitMsr", "LastPurPrc"]}
    },
    # SBODemoUS-Finance (more examples using real tables)
    {
        "db_id": "SBODemoUS-Finance",
        "question": "Show all chart of accounts with account name and current total.",
        "schema_links": {"OACT": ["AcctCode", "AcctName", "CurrTotal"]}
    },
    {
        "db_id": "SBODemoUS-Finance",
        "question": "Show journal entries with their transaction ID and memo.",
        "schema_links": {"OJDT": ["TransId", "Memo", "RefDate"]}
    },
    # SBODemoUS-Banking (more examples)
    {
        "db_id": "SBODemoUS-Banking",
        "question": "List all outgoing payments with vendor code and payment date.",
        "schema_links": {"OVPM": ["DocNum", "CardCode", "DocDate", "DocTotal"]}
    },
    # SBODemoUS-Service (more examples using real columns)
    {
        "db_id": "SBODemoUS-Service",
        "question": "Show all service contracts and their template and renewal status.",
        "schema_links": {"OCTR": ["ContractID", "CntrcTmplt", "Renewal"]}
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
