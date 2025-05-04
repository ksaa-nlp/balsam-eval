import psycopg2
from psycopg2 import sql


def export_full_database(connection_params, output_file):
    conn = None
    try:
        conn = psycopg2.connect(**connection_params)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [table[0] for table in cursor.fetchall()]

        with open(output_file, 'w') as f:
            # Export each table
            for table in tables:
                # Get table structure
                cursor.execute(sql.SQL("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = %s
                    ORDER BY ordinal_position
                """), [table])
                columns = cursor.fetchall()

                # Write CREATE TABLE statement
                f.write(f"\n-- Table: {table}\n")
                f.write(f"DROP TABLE IF EXISTS {table} CASCADE;\n")
                f.write(f"CREATE TABLE {table} (\n")

                column_defs = []
                for col in columns:
                    col_def = f"    {col[0]} {col[1]}"
                    if col[2] == 'NO':
                        col_def += " NOT NULL"
                    if col[3]:
                        col_def += f" DEFAULT {col[3]}"
                    column_defs.append(col_def)

                f.write(",\n".join(column_defs))

                # Add primary keys
                cursor.execute(sql.SQL("""
                    SELECT a.attname
                    FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                    WHERE i.indrelid = %s::regclass AND i.indisprimary
                """), [table])
                pks = [pk[0] for pk in cursor.fetchall()]
                if pks:
                    f.write(",\n    PRIMARY KEY (")
                    f.write(", ".join(pks))
                    f.write(")")

                f.write("\n);\n\n")

                # Export data
                cursor.execute(
                    sql.SQL("SELECT * FROM {}").format(sql.Identifier(table)))
                rows = cursor.fetchall()

                if rows:
                    # Get column names
                    cursor.execute(sql.SQL("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s 
                        ORDER BY ordinal_position
                    """), [table])
                    col_names = [col[0] for col in cursor.fetchall()]

                    for row in rows:
                        values = []
                        for value in row:
                            if value is None:
                                values.append("NULL")
                            elif isinstance(value, str):
                                values.append(f"'{value.replace("'", "''")}'")
                            elif isinstance(value, (bytes, bytearray)):
                                values.append(f"E'\\\\x{value.hex()}'")
                            else:
                                values.append(str(value))
                        f.write(
                            f"INSERT INTO {table} ({', '.join(col_names)}) VALUES ({', '.join(values)});\n")

        print(f"Full database exported to {output_file}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

# DATABASE_URI_DEV=postgresql://postgres:@0.0.0.0:1234/postgres


# Example usage
connection_params = {
    'host': '0.0.0.0',
    'database': 'postgres',
    'user': 'postgres',
    'password': '12345678',
    'port': 1234
}

export_full_database(connection_params, 'full_export.sql')
