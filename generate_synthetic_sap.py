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
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "What are the phone numbers and email addresses of business partners?",
        "schema_links": {"OCRD": ["CardCode", "CardName", "Phone1", "E_Mail"]}
    },
    {
        "db_id": "SBODemoUS-Business Partners",
        "question": "Show the target amount and sales employee for each business partner.",
        "schema_links": {"OCRD": ["CardCode", "SlpCode", "DfltDisc"]}
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
    # SBODemoUS-General (still failing)
    {
        "db_id": "SBODemoUS-General",
        "question": "List all business objects with their names and types.",
        "schema_links": {"OBOB": ["BisObjType", "BisObjName"]}
    },
    {
        "db_id": "SBODemoUS-General",
        "question": "Show all users with their first name, last name, and email.",
        "schema_links": {"OUSR": ["USERID", "U_NAME", "E_Mail"]}
    },
    {
        "db_id": "SBODemoUS-General",
        "question": "What are the form names and their descriptions in the system?",
        "schema_links": {"OCFM": ["FormID", "Name"]}
    },
    # SBODemoUS-Reports (still failing)
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show all queries with their category and last update date.",
        "schema_links": {"OUQR": ["QName", "QCategory", "QLastDate"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "List report documents that use email font conversion.",
        "schema_links": {"SRT2": ["DocCode", "DocName", "EmailFont", "SwpInEmail"]}
    },
    {
        "db_id": "SBODemoUS-Reports",
        "question": "Show the report wizard ID, card code, and payment number for each document.",
        "schema_links": {"RDOC": ["WizardId", "CardCode", "PmntNum"]}
    },
    # SBODemoUS-Sales Opportunities (still failing)
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "Show all opportunity stages with their name and closing percentage.",
        "schema_links": {"OOST": ["Num", "Name", "ClsPrcnt"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "List sales opportunities with their predicted closing date and sales employee.",
        "schema_links": {"OPR1": ["OpprId", "CloseDate", "SlpCode"]}
    },
    {
        "db_id": "SBODemoUS-Sales Opportunities",
        "question": "What is the total amount and currency for each sales opportunity?",
        "schema_links": {"OPR1": ["OpprId", "MaxSumLoc", "DocCurr"]}
    },
    # SBODemoUS-Inventory and Production (still failing)
    {
        "db_id": "SBODemoUS-Inventory and Production",
        "question": "Show all production orders with their planned quantity and status.",
        "schema_links": {"OWOR": ["DocNum", "PlannedQty", "Status"]}
    },
    {
        "db_id": "SBODemoUS-Inventory and Production",
        "question": "List items with their unit of measure and price.",
        "schema_links": {"OITM": ["ItemCode", "ItemName", "SalUnitMsr", "LastPurPrc"]}
    },
    {
        "db_id": "SBODemoUS-Inventory and Production",
        "question": "Show goods receipt documents with their posting date and total amount.",
        "schema_links": {"OPDN": ["DocNum", "DocDate", "DocTotal"]}
    },
    # SBODemoUS-Finance (more examples)
    {
        "db_id": "SBODemoUS-Finance",
        "question": "Show all chart of accounts with account name and account type.",
        "schema_links": {"OACT": ["AcctCode", "AcctName", "ActType"]}
    },
    {
        "db_id": "SBODemoUS-Finance",
        "question": "List all open invoices with customer code and due date.",
        "schema_links": {"OINV": ["DocNum", "CardCode", "DocDueDate", "DocTotal"]}
    },
    {
        "db_id": "SBODemoUS-Finance",
        "question": "Show journal entries with their debit and credit amounts.",
        "schema_links": {"JDT1": ["TransId", "Debit", "Credit", "Account"]}
    },
    # SBODemoUS-Banking (more examples)
    {
        "db_id": "SBODemoUS-Banking",
        "question": "List all outgoing payments with vendor code and payment date.",
        "schema_links": {"OVPM": ["DocNum", "CardCode", "DocDate", "DocTotal"]}
    },
    {
        "db_id": "SBODemoUS-Banking",
        "question": "Show checks with their account number and check amount.",
        "schema_links": {"OCHO": ["CheckKey", "AcctNum", "CheckSum"]}
    },
    # SBODemoUS-Service (more examples)
    {
        "db_id": "SBODemoUS-Service",
        "question": "List service calls with their priority and assigned technician.",
        "schema_links": {"OSCL": ["callID", "Priority", "technician"]}
    },
    {
        "db_id": "SBODemoUS-Service",
        "question": "Show all service contracts and their contract type.",
        "schema_links": {"OCTR": ["ContractID", "ContractType", "Status"]}
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
