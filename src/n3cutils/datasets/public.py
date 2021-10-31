
def sayhi():
    print("hello there")


def test_sayhi():
    sayhi()
    return True


def register_temp(df, name):
    df.createOrReplaceTempView(name)


def get_temp(name, spark):
    return spark.table(name)
