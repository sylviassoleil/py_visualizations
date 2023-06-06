import time
import datetime
import os

import requests
from data_utils.file_manipulation import create_folders
import pickle
import pandas as pd

'''
query GetUsers {
  search(
    query: "location:japan sort:followers"
    first: 25
    type: USER
  ) {
    userCount
    pageInfo {
      endCursor
      hasNextPage
    }
    edges {
      node {
        ... on User {
          email
          login
          name
          location
          url
          followers {
            totalCount
          }
          contributionsCollection(from:"2018-01-01T00:00:00", to:"2018-12-31T00:00:00") {
            totalPullRequestContributions
            totalIssueContributions
            totalCommitContributions
            totalRepositoryContributions
#             commitContributionsByRepository
#             {
#               contributions {
#                 totalCount
#               }
#             }
#             totalPullRequestContributions
#              {
#               contributionMonths
#               totalContributions
#             }
          }
          avatarUrl
        }
      }
    }
  }
}
'''


def format_query(followers_cond, starting_alphabet, order, cursor):
    graphql_stat = '''
query getUsersOrganizations {
  search(
    query: "location:japan followers:%s %s sort:followers%s"
    first: 100
    type: USER
    %s
  ) {
    userCount
    pageInfo {
      endCursor
      hasNextPage
    }
    edges {
      node {
        ... on User {
         login
         followers {
        totalCount
    }
         organizations(first: 100) {
           edges {
             node { 
                id
                login
                location
              }
          }
          }
        }
      }
    ''' % (followers_cond, starting_alphabet, order, cursor)
    return graphql_stat

def format_org_query(created_date, cursor):
    # todo change it to 100
    graphql_stat = '''
    query getOrganizations {
              search(
                query: "type:org location:japan created:%s sort:createdAt-asc"
                first: 100
                type: USER
                %s
                ) {
                userCount
                pageInfo {
                  endCursor
                  hasNextPage
                }
                edges { 
                  node {
                    ... on Organization {
                     login
                     location
                     createdAt
                    }
                  
                }
                  }
                }
      }
    ''' % (created_date, cursor)
    return graphql_stat

class APIConfig:
    name = "sylviassoleil"
    api_key = "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"

class Folders:
    project_name = 'jp_github_org'
    cwd = os.getcwd()
    base_folder = os.path.join(cwd, 'raw_data')
    base_procesed_folder = os.path.join(cwd, 'processed_data')
    raw_folder = os.path.join(base_folder, project_name)
    processed_folder = os.path.join(base_procesed_folder, project_name)


def run_query(graphql_stat):  # A simple function to use requests.post to make the API call. Note the json= section.
    headers = {"Authorization": "Bearer %s" % APIConfig.api_key}
    request = requests.post('https://api.github.com/graphql',
                            json={'query': graphql_stat}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, graphql_stat))

def dump_file(df, cursor_name):
    if not cursor_name:
        cursor_name = '0'
    file_name = os.path.join(Folders.raw_folder, cursor_name)
    with open(file_name, 'wb') as w:
        pickle.dump(df, w)

def get_last_item_followercount(df):
    fol_count = 0
    ed = df['data']['search']['edges']
    if ed:
        if 'node' in ed:
            ed = ed['node']
            if 'followers' in ed:
                fol_count = ed['followers']['totalCount']
    return fol_count

def get_last_created_date(df):
    ed = df['data']['search']['edges']
    if ed:
        dates = sorted([i['node']['createdAt'] for i in ed], reverse=True)
        return dates[0]
        # ed = ed[-1]
        # if 'node' in ed:
        #     ed = ed['node']
        #     return ed.get('createdAt')


def request_data(followers_cond, starting_alphabet, order='', cursor='', counter=0):
    query = format_query(followers_cond, starting_alphabet, order, cursor)
    try:
        df = run_query(query)
    except:
        time.sleep(30*60)
    time.sleep(5)

    page_info = df['data']['search']['pageInfo']
    if page_info['hasNextPage']:
        cursor_string = page_info['endCursor']
        dump_file(df, cursor_string.replace(os.sep, '')+str(datetime.datetime.utcnow()))
        cursor = 'after: "%s"' % cursor_string
        counter+=1
        print(starting_alphabet, counter*100)
        return request_data(followers_cond, starting_alphabet, order, cursor, counter)
    else:
        follower_cnt = get_last_item_followercount(df)
        # df['data']['search']['edges'][-1]['node']['followers']['totalCount']
        dump_file(df, cursor.replace(os.sep, '')+str(datetime.datetime.utcnow()))
        return follower_cnt

def request_org_dynamic_queue():
    time_format = '%Y-%m-%dT%H:%M:%SZ'
    start_date = '<2011-01-01T00:00:00Z'
    end_date = '2023-02-01'
    start_date = request_org_data(start_date)
    # sorting does not really work so this doesn't cover the full dataset
    while start_date is not None:
        start_date = request_org_data(f'>{start_date}')
        if start_date > end_date:
            break

def request_org():
    time_format = '%Y-%m-%dT%H:%M:%SZ'
    step = 30
    start_date = pd.to_datetime('2011-02-01', utc=True)
    cutoff_date = pd.to_datetime('2023-01-01', utc=True)
    while start_date < cutoff_date:
        end_date = start_date+pd.DateOffset(days=step)
        date_str = '%s..%s' % (start_date.strftime(time_format), end_date.strftime(time_format))

        end_ = request_org_data(date_str)
        start_date = end_date
        if end_ is not None:
            if pd.to_datetime(end_)>start_date:
                start_date = pd.to_datetime(end_)


def request_org_data(created_date, cursor='', counter=0):
    '''
    created_Date: the range of
    created:2020-01-01T00:00:00..2020-02-01T00:00:00
    '''
    query = format_org_query(created_date, cursor)
    try:
        df = run_query(query)
    except:
        time.sleep(30*60)
    time.sleep(5)

    page_info = df['data']['search']['pageInfo']
    if page_info['hasNextPage']:
        cursor_string = page_info['endCursor']
        dump_file(df, cursor_string.replace(os.sep, '')+str(datetime.datetime.utcnow()))
        cursor = 'after: "%s"' % cursor_string
        counter+=1
        print(created_date, counter*100)
        return request_org_data(created_date, cursor, counter)
    else:
        # total = df['data']['search']['userCount']
        # df['data']['search']['edges'][-1]['node']['followers']['totalCount']
        dump_file(df, cursor.replace(os.sep, '')+str(datetime.datetime.utcnow()))
        created_at = get_last_created_date(df)
        return created_at

def process_org():
    files = [os.path.join(Folders.raw_folder, f) for f in os.listdir(Folders.raw_folder)]
    output = [pd.DataFrame(pd.read_pickle(f)['data']['search']['edges']) for f in files]
    output = pd.concat(output)
    output = output['node'].apply(pd.Series)
    output = output.drop_duplicates(subset=['login'])
    output_file = os.path.join(Folders.processed_folder, 'org.csv')
    output.to_csv(output_file, index=False)
def collect_org():
    files = [os.path.join(Folders.raw_folder, f) for f in os.listdir(Folders.raw_folder)]
    output = []
    for f in files:
        df = pd.read_pickle(f)
        data = df['data']['search']['edges']
        org = list(filter(lambda i: pd.notna(i), [i['node'].get('organizations') for i in data]))
        if org:
            org = pd.DataFrame(org)
            output.append(org)
            # org = org['edges'].apply(pd.Series)
    output = pd.concat(output)
    output_df = output.explode('edges')
    output_df.dropna(inplace=True)
    output_df = output_df['edges'].apply(lambda i: pd.Series(i['node']))
    output_df.drop_duplicates(subset=['id'], inplace=True)
    country_locations = '/Users/sylvia/PycharmProjects/dev_projects/japan_locations.json'
    locations = pd.read_json(country_locations)
    locations = locations[0].map(lambda i: i.lower().strip()).tolist()
    is_jp = lambda i: any([l in (i.lower().strip() if pd.notna(i) else '') for l in locations])
    output_jp = output_df[output_df['location'].map(is_jp)]
    return ','.join(list(map(lambda i: f'"{i}"', output_jp.login.tolist())))

def developers():
    files = [os.path.join(Folders.raw_folder, f) for f in os.listdir(Folders.raw_folder)]
    output = []
    for f in files:
        df = pd.read_pickle(f)
        eg = df['data']['search']['edges']
        data = pd.DataFrame(eg)
        if not 'node' in data.columns:
            continue
        data = data['node'].apply(pd.Series)
        data['followers'] = data['followers'].map(lambda i: i.get('totalCount') if type(i)==dict else None)
        output.append(data[['login', 'followers']])
    df = pd.concat(output)
    df.dropna(subset=['login'], inplace=True)
    df.drop_duplicates(subset=['login'], inplace=True)


def request_users_jp():
    from string import ascii_lowercase
    dec_followers_cond = '>1'
    last_follower_cnt = request_data(dec_followers_cond, starting_alphabet='')
    last_follower_cnt_record = []
    while last_follower_cnt > 0:
        last_follower_cnt = request_data(f'<{last_follower_cnt}', starting_alphabet='')
        if last_follower_cnt in last_follower_cnt_record:
            break
        last_follower_cnt_record.append(last_follower_cnt)

    fn = 'fullname:{}'
    for s in ascii_lowercase:
        for i in last_follower_cnt_record:
            dec_followers_cond = f'={i}'
            request_data(dec_followers_cond, fn.format(s))

        dec_followers_cond = '<{}'.format(last_follower_cnt_record[-1])
        request_data(dec_followers_cond, fn.format(s))
        time.sleep(15)
        # asc_followers_cond = '<2'
        request_data(dec_followers_cond, fn.format(s), '-asc')
        time.sleep(15)

if __name__ == '__main__':
    pass
    create_folders(Folders)
    request_org()
    # import datetime
    # n = datetime.datetime.utcnow()
    # # create at
    # language
    # type:org location:japan created_at



    # api_key = {"sylviassoleil": "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"}
    # headers = {"Authorization": f"Bearer {api_key['sylviassoleil']}"}
    # request = requests.post('https://api.github.com/graphql', json={'query': graphql_stat}, headers=headers)
    # df = request.json()
    # d = pd.read_pickle(os.path.join(Folders.raw_folder, '2023-02-07 05:15:42.023813'))


