#!/usr/bin/env python3

from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

zk_cluster = "127.0.0.1:2181"
zk_path = "/openmldb"

db = SQLDatabase.from_uri(f"openmldb:///db1?zk={zk_cluster}&zkPath={zk_path}")

llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, top_k=1)

#db_chain.run("How many employees are there?")
#db_chain.run("create database which is named db2")
db_chain.run("create table which has one column named id and its type is int")
