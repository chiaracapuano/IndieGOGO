class Create_set:
    def __init__(self, engine):
        self.engine = engine

    def maskunion(self):

        con = self.engine.raw_connection()
        cursor = con.cursor()
        cursor.callproc('maskunion', ['public', 'TF-IDF_ml_set','IDF_ml_set_complete'])
        con.commit()




