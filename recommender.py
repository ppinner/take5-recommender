# importing necessary libraries
import pandas as pd
import datetime as dt
from pymongo import MongoClient
from bson.objectid import ObjectId
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

password = "dataGuy1"
username = "dataGuy"
db = "Take5"
host = 'cluster0.uyjm9.mongodb.net'


def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb+srv://dataGuy:dataGuy1@cluster0.uyjm9.mongodb.net'
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]


def read_mongo(db, collection, query, host, port, username, password, no_id, agg):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    if agg:
        cursor = db[collection].aggregate(query)
    else:
        cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df = pd.DataFrame(iter(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df


def format_user_dataframe(user_dataframe):
    df = pd.json_normalize(user_dataframe['personality'])
    output_df = pd.DataFrame.from_dict(df)
    output_df['dob'] = pd.to_numeric(pd.to_datetime(user_dataframe['dob']))
    output_df['_id'] = user_dataframe['_id']

    return output_df


def get_similar_users(user_id, num_neighbours):
    # Step 1: Get users and find alike users to current user
    all_users = read_mongo(db, 'users', {}, host, 27018, username, password, False, False)
    userDF = format_user_dataframe(all_users).fillna(0)
    pivot_user_based = pd.pivot_table(userDF, index='_id').T
    sparse_pivot = sparse.csr_matrix(pivot_user_based.fillna(0))

    similar_users = cosine_similarity(sparse_pivot)
    users_df = pd.DataFrame(similar_users,
                            columns=pivot_user_based.columns,
                            index=pivot_user_based.columns)
    # remove current user from neighbours
    users_df = users_df.drop(ObjectId(user_id))

    usr_cosine_df = pd.DataFrame(users_df[ObjectId(user_id)].sort_values(ascending=False))
    usr_cosine_df.reset_index(level=0, inplace=True)
    usr_cosine_df.columns = ['_id', 'cosine_sim']
    return usr_cosine_df.head(num_neighbours)['_id'].tolist()


def get_user_activities(users):
    oids = []
    for u in users:
        oids.append(str(u))

    pipeline = [
        {"$match": {"userId": {"$in": oids}}},
        {"$unwind": {"path": '$activity'}},
        {"$project": {"logId": "$_id", "userId": "$userId", "activityId": "$activity._id"}},
        {"$group": {"_id": "$activityId", "times_logged": {"$sum": 1}}},
        {"$sort": {"times_logged": -1}}
    ]

    activityLogs = read_mongo('Take5', 'activityLogs', pipeline, host, 27018, username, password, False, True)

    return activityLogs


def filter_activities(activities, user_id):
    # get user
    this_user = read_mongo(db, 'users', {"_id": ObjectId(user_id)}, host, 27018, username,
                           password, False, False)

    focus = this_user["focus"][0]
    # TODO - get to incorporate lowest scoring goal also
    # lowestScore = this_user["scores"].last()

    oids = []
    for a in activities["_id"]:
        oids.append(ObjectId(a))
    activityOptions = read_mongo(db, 'activities', {"_id": {"$in": oids}}, host, 27018, username,
                                 password, False, False)

    categoriesExpanded = pd.DataFrame(activityOptions['category'].tolist()).add_prefix('category')
    activityOptions[categoriesExpanded.columns.values] = categoriesExpanded
    activityOptions = activityOptions.drop('category', axis=1)
    merged = activityOptions.merge(activities, on="_id")

    # get only those in user focus and lowest scoring category
    # filtered = merged[merged[categoriesExpanded.columns.values].str.contains(focus)]
    queryString = filterByCategoryStringBuilder(categoriesExpanded.columns.values, focus)
    filtered = merged.query(queryString)
    return filtered


def filterByCategoryStringBuilder(categoryHeaders, focus):
    string = ""
    i = 0
    for cat in categoryHeaders:
        string += cat + " == '" + focus + "'"
        if i < len(categoryHeaders) - 1:
            string += " | "
        i += 1

    return string


def recommend_for_user(user_id):
    similar_users = get_similar_users(user_id, 5)

    # Step 2: From set of similar users (max 10), find most popular activity IDs
    activities = get_user_activities(similar_users)

    # Step 3: Filter these activities by the user's current goal/lowest score and return top suggestion
    if len(activities) > 1:
        filtered_activities = filter_activities(activities, user_id)
        return filtered_activities.nlargest(1, 'times_logged')
    elif len(activities) == 0:
        return None
    else:
        return activities.nlargest(1, 'times_logged')


print(recommend_for_user("61cc7e34b137a57798047db1"))
