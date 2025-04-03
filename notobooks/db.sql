-- Create the CUSTOMER table
CREATE TABLE CUSTOMER (
    CUS_CODE INT PRIMARY KEY,
    CUS_LNAME VARCHAR2(50),
    CUS_FNAME VARCHAR2(50),
    CUS_INITIAL CHAR(1),
    CUS_AREACODE CHAR(3),
    CUS_PHONE CHAR(8),
    CUS_CREDITLIMIT DECIMAL(10, 2),
    CUS_BALANCE DECIMAL(10, 2),
    CUS_DATELSTPMT DATE,
    CUS_DATELSTPUR DATE
);

-- INVOICE table
CREATE TABLE INVOICE (
    INV_NUMBER INT PRIMARY KEY,
    CUS_CODE INT,
    INV_DATE DATE,
    INV_TOTAL DECIMAL(10, 2),
    INV_TERMS VARCHAR2(50),
    INV_STATUS VARCHAR2(20),
    FOREIGN KEY (CUS_CODE) REFERENCES CUSTOMER(CUS_CODE)
);
--  PAYMENTS table
CREATE TABLE PAYMENTS (
    PMT_ID INT PRIMARY KEY,
    PMT_DATE DATE,
    CUS_CODE INT,
    PMT_AMT DECIMAL(10, 2),
    PMT_TYPE VARCHAR2(20),
    PMT_DETAILS VARCHAR2(100),
    FOREIGN KEY (CUS_CODE) REFERENCES CUSTOMER(CUS_CODE)
);


--  VENDOR table
CREATE TABLE VENDOR (
    V_CODE INT PRIMARY KEY,
    V_NAME VARCHAR2(50),
    V_CONTACT VARCHAR2(50),
    V_AREACODE CHAR(3),
    V_PHONE CHAR(8),
    V_STATE VARCHAR2(20),
    V_ORDER INT
);
--  PRODUCT table
CREATE TABLE PRODUCT (
    P_CODE INT PRIMARY KEY,
    P_DESCRIPT VARCHAR2(100),
    P_INDATE DATE,
    P_QTYOH INT,
    P_MIN INT,
    P_PRICE DECIMAL(10, 2),
    P_DISCOUNT DECIMAL(5, 2),
    V_CODE INT,
    FOREIGN KEY (V_CODE) REFERENCES VENDOR(V_CODE)
);
--  LINE table
CREATE TABLE LINE (
    INV_NUMBER INT,
    LINE_NUMBER INT,
    P_CODE INT,
    LINE_UNITS INT,
    LINE_PRICE DECIMAL(10, 2),
    PRIMARY KEY (INV_NUMBER, LINE_NUMBER),
    FOREIGN KEY (INV_NUMBER) REFERENCES INVOICE(INV_NUMBER),
    FOREIGN KEY (P_CODE) REFERENCES PRODUCT(P_CODE)
);



/* Transaction Management (chapter 10)
ABC Markets sells products to customers. The relational ER diagram shown in Figure represents the main entities for ABC’s database. Note the following important characteristics:
•	A customer may make many purchases, each one represented by an invoice. 
		The CUS_BALANCE is updated with each credit purchase or payment and represents the amount the customer owes.
		The CUS_BALANCE is increased (+) with every credit purchase and decreased (-) with every customer payment.
		The date of last purchase is updated with each new purchase made by the customer.
		The date of last payment is updated with each new payment made by the customer. 
•	An invoice represents a product purchase by a customer. 
		An INVOICE can have many invoice LINEs, one for each product purchased. 
		The INV_TOTAL represents the total cost of invoice including taxes.
		The INV_TERMS can be “30,” “60,” or “90” (representing the number of days of credit) or “CASH,” “CHECK,” or “CC.”
		The invoice status can be “OPEN,” “PAID,” or “CANCEL.”
•	A product’s quantity on hand (P_QTYOH) is updated (decreased) with each product sale.
•	A customer may make many payments. The payment type (PMT_TYPE) can be one of the following:
		“CASH” for cash payments.
		“CHECK” for check payments
		“CC” for credit card payments
•	The payment details (PMT_DETAILS) are used to record data about check or credit card payments:
		The bank, account number, and check number for check payments
		The issuer, credit card number, and expiration date for credit card payments.


-- ANSWERS
/* a.	On May 11, 2016, customer ‘10010’ makes a credit purchase (30 days) of one unit of product ‘11QER/31’ 
with a unit price of $110.00; the tax rate is 8 percent.The invoice number is 10983,
and this invoice has only one product line.*/

BEGIN TRANSACTION;

INSERT INTO INVOICE (INV_NUMBER, CUS_CODE, INV_DATE, INV_TOTAL, INV_TERMS, INV_STATUS)
    VALUES (10983, 10010, '05/11/2016', 110.00 * 1.08, '30', 'OPEN');

INSERT INTO LINE (INV_NUMBER, LINE_NUMBER, P_CODE, LINE_UNITS, LINE_PRICE) 
	VALUES (10983, 1, '11QER/31', 1, 110.00);

UPDATE CUSTOMER
    SET CUS_BALANCE = CUS_BALANCE + 110.00*1.08,
		CUS_DATELSTPUR = '05/11/2016'
    WHERE CUS_CODE = 10010;

UPDATE PRODUCT
    SET P_QTYOH = P_QTYOH - 1
    WHERE P_CODE = '11QER/31';
COMMIT
/*@Delphin Juma Kaduli*/


/* b.	On June 3, 2016, customer ‘10010’ makes a payment of $100 in cash.   
The payment ID is 3428.*/

BEGIN TRANSACTION;

INSERT INTO PAYMENTS (PMT_ID, PMT_DATE, CUS_CODE, PMT_AMT, PMT_TYPE, PMT_DETAILS)
VALUES (3428, '06/03/2016', 10010, 100.00, 'CASH', NULL);
	
UPDATE CUSTOMER
    SET CUS_BALANCE = CUS_BALANCE - 100.00,
		CUS_DATELSTPMT ='06/03/2016'
    WHERE CUS_CODE = 10010;
	
COMMIT;
/*@Delphin Juma Kaduli*/