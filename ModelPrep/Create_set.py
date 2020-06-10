class Create_set:
    def __init__(self, engine):
        self.engine = engine

    def maskunion(self):

        con = self.engine.raw_connection()
        cursor = con.cursor()
        cursor.callproc('maskunion', ['public', 'sorted_ml_set_2016','ml_set_complete'])
        con.commit()




