from google_sheet_utils import GoogleSheetConnector
from config import gs_key_path
import os
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")
import matplotlib.pyplot as plt

gs_reader = GoogleSheetConnector(gs_key_path)
gs_id = 'XXXX'
img_folder = 'github_png'


def get_data():
    df = gs_reader.get_data_by_sheet_id(gs_id, 'bq')
    df['first_event_date'] = pd.to_datetime(df['first_event_date'])
    df_created = df.groupby('login').first_event_date.min().reset_index()
    df_created = df_created.sort_values(by=['first_event_date'], ascending=False)
    df_created['github_url'] = 'https://github.com/' + df_created['login']
    df_created['first_event_date'] = df_created['first_event_date'].astype(str)
    df_created['first_date'] = df_created['first_event_date'].map(lambda i: int(i[:4]))
    df_year = df.groupby('year').login.nunique().reset_index()
    gs_reader.write_into_worksheet(df_year, gs_id, 'org_year')
    gs_reader.write_into_worksheet(df_created, gs_id, 'org_first_event')


def process_org_repo():
    folder = 'adhoc_data/github_biquery'
    process_folder = 'processed_data/jp_github_org'
    repo = pd.read_csv(os.path.join(folder, 'org_repos.csv'))

    org_year = repo.groupby(['year', 'login']).agg(repo=('id', 'nunique')).reset_index()
    ridge_density(org_year.query('repo<8000'), hue_var='year', x_var='repo', x_label='# Repos (log scaled)',
                  title='Distribution of Active Repo Count per Organization', log_scale=True, clip_on=True)

    ''' stack of repo count  '''
    # cutoffs = np.percentile(org_year.query('year==15').repo_count, [0.25, 0.5, 0.75])
    org_year = org_year.query('repo<8000')
    d = org_year.groupby('year').repo.describe(percentiles=[.25, .5, .75, .80, .85, .90, .95, .98, .99]).reset_index()
    tab_name = 'new_repo_descriptive'
    gs_reader.del_worksheet(gs_id, [tab_name])
    gs_reader.write_into_worksheet(d, gs_id, tab_name)

    org_year['first_year'] = org_year.groupby('login').year.transform('min')
    org_year['age'] = org_year['year'].astype(int) - org_year['first_year'].astype(int)

    ''' age & productivity '''
    age_df = org_year.query('(repo<8000)&(year>15)').groupby('age').repo.describe(percentiles=[.25, .5, .75, .80, .85, .90, .95, .98, .99]).reset_index()
    ridge_density(org_year.query('(repo<8000)&(year>15)'), hue_var='age', x_var='repo', x_label='# Repos (log scaled)',
                  title='Distribution of Active Repo Count per Organization by Ages', log_scale=True, clip_on=True)
    tab_name = 'repo by age'
    gs_reader.write_into_worksheet(age_df, gs_id, tab_name)
    color_palette_ = sns.color_palette("Spectral", as_cmap=False, n_colors=8)
    color_palette = {k: v for k, v in zip(repo.year.unique(), color_palette_)}
    ''' heatmap of orgs'''
    heatmap_df(org_year)
    ''' stacked plot '''
    cutoff = [1, 3, 7, 14, 40, 100]
    # 25, 50, 75, 90, 99,
    org_label = org_year.query('(login!="kanripo")|(year>2016)')
    # c1 = org_label['repo']<=1
    c2 = (org_label['repo'] > 1) & (org_label['repo'] <= 3)
    c3 = (org_label['repo'] > 3) & (org_label['repo'] <= 7)
    c4 = (org_label['repo'] > 7) & (org_label['repo'] <= 14)
    c5 = (org_label['repo'] > 14) & (org_label['repo'] <= 40)
    c6 = (org_label['repo'] > 40) & (org_label['repo'] <= 100)
    c7 = (org_label['repo'] > 100)
    conditions = [c2, c3, c4, c5, c6, c7]
    label_names = ['2-3', '4-7', '8-14', '15-40', '41-100', '>100']
    org_label['label'] = np.select(conditions, label_names, '1')
    repo_org_label = org_label.copy()

    org_label = org_label.groupby(['year', 'label']).login.nunique().reset_index()
    org_label['pct'] = org_label['login'] / org_label.groupby('year').login.transform('sum')

    series_name = ['1', *label_names]
    org_label = pd.pivot_table(org_label, index=['label'], columns=['year'], values=['login', 'pct'])
    org_label = org_label[['pct']].transpose().reset_index()[['year', *series_name]]
    org_label = org_label.set_index('year')
    org_label = org_label * 100

    series_name = ['1', *label_names]
    stacked_bar(org_label, '% Breakdown of Organizatons by Organization Project Size', 'year', 'pct', label_series=series_name)

    ''' repo pct '''
    repo_org_label = repo_org_label.groupby(['year', 'label']).repo.sum().reset_index()
    repo_org_label['pct'] = repo_org_label['repo'] / repo_org_label.groupby('year').repo.transform('sum')
    series_name = ['1', *label_names]
    repo_org_label = pd.pivot_table(repo_org_label, index=['label'], columns=['year'], values=['repo', 'pct'])
    repo_org_label = repo_org_label[['pct']].transpose().reset_index()[['year', *series_name]]
    repo_org_label = repo_org_label.set_index('year')
    repo_org_label = repo_org_label * 100

    series_name = ['1', *label_names]
    stacked_bar(repo_org_label, "% Breakdown of # Repositories by Organization Project Size", 'year', 'pct', label_series=series_name)


    ''' org yoy v.s. repo yoy'''
    repo_ = repo.query('login!="kanripo"|year>16')
    org_repo = repo_.groupby('year').agg(orgs=('login', 'nunique'), repos=('id', 'nunique')).reset_index()
    org_repo = org_repo.sort_values(by='year', ascending=True)
    org_repo['orgs_yoy'] = org_repo.orgs.pct_change() * 100
    org_repo['repos_yoy'] = org_repo.repos.pct_change() * 100
    org_repo_chg = org_repo[['year', 'orgs_yoy', 'repos_yoy']].melt(id_vars=['year'],
                                                                    value_vars=['orgs_yoy', 'repos_yoy'],
                                                                    var_name='type', value_name='YoY')
    org_repo_chg['YoY'] = org_repo_chg['YoY']
    single_axis_chart(org_repo_chg, 'Active Orgs YOY % v.s. Active Repo YoY %', x_var='year', y_var='YoY',
                      hue_var='type', figsize=(30, 12))

    ''' org_count '''
    double_axis_bar_line_plot(org_repo[['year', 'orgs']], org_repo[['year', 'orgs_yoy']], x_var='year', y_bar='orgs',
                              y_line='orgs_yoy',
                              title='Total Active Organizations')

    double_axis_bar_line_plot(org_repo[['year', 'repos']], org_repo[['year', 'repos_yoy']], x_var='year', y_bar='repos',
                              y_line='repos_yoy',
                              title='Total Active Repos')

    ''' retention type '''

    date_var = 'year'
    reindex_array = list(range(org_year[date_var].min(), org_year[date_var].max() + 1))
    type_df = find_entity_type_turnover(org_year, date_var, user_id='login', label_var='type',
                                        reindex_array=reindex_array)
    title, x_var, y_var = 'Organization Growth Accounting', 'year', '# of Organizations'
    user_type_chart_single_series(type_df, title, x_var, y_var, label_size=18)

    '''repo'''
    repo_type_df = find_entity_type_turnover(repo_, date_var, user_id='id', label_var='type',
                                        reindex_array=reindex_array)
    title, x_var, y_var = 'Repo Growth Accounting', 'year', '# of Repos'
    user_type_chart_single_series(repo_type_df, title, x_var, y_var, label_size=18)
    return


def user_type_chart_single_series(df, title, x_var, y_var, label_size=14):
    ''' input
    :param df colummns = ['churned', 'retained', 'resurrected', 'new']

    '''
    color_palette = {'churned': '#3F681C',
                     'new': '#FFBB00',
                     'resurrected': '#FB6542',
                     'retained': '#375E97'}
    labels = list(color_palette.keys())
    # df['resurrected'] = np.where(df['resurrected'].isna(), None, df['new'].fillna(0)+df['resurrected'])
    # df['retained'] = np.where(df['retained'].isna(), None, df['retained']+df['resurrected'].fillna(0))

    if set(labels) - set(df.columns):
        raise ValueError("%s columns don't meet input requiremnts of including %s" % (df.columns, labels))
    plt.close()
    fig, ax1 = plt.subplots(figsize=(30, 12))

    df.set_index('year')[labels].plot(kind='bar', stacked=True, color=color_palette, ax=ax1, legend=False)
    handles, labels = plt.gca().get_legend_handles_labels()

    for c in ax1.containers:
        val_labels = ['{:g}'.format(abs(v.get_height())) if v.get_height()!=0 else '' for v in c]
        # remove the labels parameter if it's not needed for customized labels
        ax1.bar_label(c, labels=val_labels, label_type='center', fmt='g', padding=5, fontsize=label_size, c='w')

    df_ratio = df.copy()
    df_ratio['(new+resurrected)/churned'] = (df_ratio.new + df_ratio.resurrected) / df_ratio.churned*-1
    df_ratio.plot(y='(new+resurrected)/churned', color='#737373', ax=ax1, secondary_y=True, linewidth=3.5, legend=False,
                  fontsize=25, grid=False)
    ax1.grid(True, which='major', linewidth=2)

    ax1.set_ylabel(y_var, fontsize=25)
    ax1.set_xlabel(x_var, fontsize=25)

    # dates_val = [i.date() for i in pd.date_range(df.created_at_month.min(),
    #                                              df.created_at_month.max(), freq='3M')]
    # dates_tick = [i.strftime('%Y/%m') if i in dates_val else ''
    #               for i in df.created_at_month]
    # ax1.set_xticklabels(dates_tick)
    ax1.figure.autofmt_xdate(rotation=0, ha='center')
    ax1.yaxis.set_tick_params(labelsize=25)
    ax1.xaxis.set_tick_params(labelsize=25)
    ''' legend '''
    ax2 = ax1.twinx()
    ax2.grid(color='#E7E7EF', linewidth=0.2)
    ax2.set_yticklabels([])
    ax2.yaxis.set_tick_params(labelsize=0, color='#FFFFFF')
    ax2.set_ylabel('(New+Resurrected)/Churned', color='grey', fontsize=25, labelpad=45)

    plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.19, 1.0), fontsize=25)
    plt.title(title, fontsize=30)
    plt.tight_layout()
    plt.show()
    output_file = os.path.join(img_folder, f'{title}.png')
    fig.savefig(output_file)


def find_entity_type_turnover(user_df, date_col, user_id, label_var, reindex_array=None, freq=None):
    user_df = user_df.sort_values(by=[date_col], ascending=True).drop_duplicates(subset=[user_id, date_col])
    user_df['val'] = 1
    user_df_pv = pd.pivot_table(user_df, index=[date_col], columns=[user_id], values='val')

    ''' reindex for fill all relevant dates '''
    if reindex_array is not None:
        dates_t = reindex_array
    elif freq is not None:
        dates_t = pd.date_range(user_df_pv.index.min(), user_df_pv.index.max(), freq=freq)
    else:
        raise ValueError('can not reindex date index')
    user_df_pv = user_df_pv.reindex(dates_t).fillna(0)

    ''' get new users '''
    user_new = user_df_pv.cumsum().cumsum()
    # equivalent of new = user_df.groupby([user_id])[date_col].min().reset_index().rename(
    #     columns={date_col: 'first_month'})
    user_new = pd.melt(user_new, ignore_index=False, value_name='value').reset_index().query('value==1')
    user_new[label_var] = 'new'

    ''' find all possible retained '''
    user_retained = pd.melt(user_df_pv, ignore_index=False).reset_index().merge(user_new[[user_id, date_col]],
                                                                                on=[user_id],
                                                                                how='left',
                                                                                suffixes=('', '_new')).query(
        f'{date_col}>{date_col}_new')

    ''' resurrected & churned '''
    user_df_pv_shift = user_df_pv.shift(1)
    user_df_pv_type = user_df_pv - user_df_pv_shift
    # -1 churned, 1 resurected, 0 & pv=1 retained
    user_df_pv_type = pd.melt(user_df_pv_type, ignore_index=False).reset_index()
    user_types = user_df_pv_type.merge(user_new[[user_id, date_col]], on=[user_id],
                                       how='left', suffixes=('', '_new')).query(f'{date_col}>{date_col}_new')[
        [user_id, date_col, 'value']]

    user_types = user_types.merge(user_retained[[user_id, date_col, 'value']], on=[user_id, date_col], how='left',
                                  suffixes=('', '_retained'))
    conditions = [user_types['value'] == -1, user_types['value'] == 1,
                  (user_types['value'] == 0) & (user_types['value_retained'] == 1)]
    types = ['churned', 'resurrected', 'retained']
    user_types[label_var] = np.select(conditions, types, None)

    ''' combine new and the other types '''
    output_cols = [user_id, date_col, label_var]
    user_types_final = pd.concat([user_types[output_cols].dropna(subset=label_var), user_new[output_cols]])

    user_types = user_types_final.groupby([date_col, label_var])[user_id].nunique().reset_index()
    user_types = pd.pivot_table(user_types, index=[date_col], columns=[label_var], values=user_id,
                               ).reset_index()
    user_types['churned'] = -user_types['churned']
    return user_types


def prepare_cdf(df, var_name, date_col):
    df = df.sort_values(by=[date_col, var_name], ascending=[True, True])
    df['percentage'] = df[var_name] / df.groupby(date_col)[var_name].transform('sum')
    # df['cumulative_pct'] = df.groupby('year').percentage.cumsum() * 100
    df['rank'] = 1
    df['rank'] = df.groupby('year')['rank'].cumsum()
    df['cdf'] = df['rank'] / df.groupby('year')['rank'].transform('max')

    df = df.sort_values(by=[date_col, var_name], ascending=[True, False])
    df['cumulative_pct'] = df.groupby('year').percentage.cumsum() * 100
    df['rank_'] = 1
    df['rank_'] = df.groupby('year')['rank_'].cumsum()

    return df


def prepare_cdf_per_age_group(df, var_name, date_col, group_key):
    group_keys = [date_col, group_key]
    df = df.sort_values(by=[date_col, group_key, var_name], ascending=[True, True, True])
    df['percentage'] = df[var_name] / df.groupby(group_keys)[var_name].transform('sum')
    # df['cumulative_pct'] = df.groupby('year').percentage.cumsum() * 100
    df['rank'] = 1
    df['rank'] = df.groupby(group_keys)['rank'].cumsum()
    df['cdf'] = df['rank'] / df.groupby(group_keys)['rank'].transform('max')
    # df = df.sort_values(by=[date_col,group_key, var_name], ascending=[True, False])
    # df['cumulative_pct'] = df.groupby('year').percentage.cumsum() * 100
    # df['rank_'] = 1
    # df['rank_'] = df.groupby('year')['rank_'].cumsum()

    return df


def chart_process():
    folder = 'adhoc_data/github_biquery'
    process_folder = 'processed_data/jp_github_org'

    org = pd.read_csv(os.path.join(process_folder, 'org.csv'))
    repo_ = pd.read_csv(os.path.join(folder, 'org_repo_count.csv'))
    repo = pd.read_csv(os.path.join(folder, 'org_repo_activity.csv'))
    repo = pd.concat([repo_, repo])

    repo = prepare_cdf(repo, 'repo_count', 'year')

    repo['first_year'] = repo.groupby('login').year.transform('min')
    repo['age'] = repo['year'].astype(int) - repo['first_year'].astype(int)

    color_palette_ = sns.color_palette("Spectral", as_cmap=False, n_colors=8)
    color_palette = {k: v for k, v in zip(repo.year.unique(), color_palette_)}
    ''' heatmap of orgs'''
    heatmap_df(repo)

    ''' retention breakdown '''
    # todo of orgs
    # of repo

    ''' histplot '''  # todo stack bar chart
    cutoffs = np.percentile(repo.query('year==15').repo_count, [0.25, 0.5, 0.75])

    ''' cdf '''
    title = 'CDF of # Repos of Organizations'
    y_var = 'cdf'
    text_condition = 'cdf==1'
    repo['repo_count_log'] = repo['repo_count'].map(lambda i: np.log(i))
    single_axis_chart(repo, title, x_var='repo_count', y_var=y_var, hue_var='year',
                      color_palette=color_palette, y_lim=[0.98, 1], x_lim=[1, 100],
                      text_condition=text_condition, log_transformed=True)


def ridge_density(df, hue_var, x_var, x_label, title, log_scale=True, clip_on=False):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create the data
    # rs = np.random.RandomState(1979)
    # x = rs.randn(500)

    # g = np.tile(list("ABCDEFGHIJ"), 50)
    # df = pd.DataFrame(dict(x=x, g=g))
    # m = df.g.map(ord)
    # df["x"] += m

    # Initialize the FacetGrid object
    fig, ax1 = plt.subplots(figsize=(15, 15))
    n_series = df[hue_var].nunique()

    pal = sns.cubehelix_palette(n_series, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row=hue_var, hue=hue_var, aspect=15, height=.6, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, x_var,
          bw_adjust=.2, clip_on=clip_on,
          fill=True, alpha=1, linewidth=2, log_scale=log_scale, ax=ax1)
    g.map(sns.kdeplot, x_var, clip_on=clip_on, color='w', lw=1, bw_adjust=.2, log_scale=log_scale, ax=ax1)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, x_var)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.fig.suptitle(title)
    # plt.title(title, fontdict={'y':1.1})
    g.set_axis_labels(x_label)

    output_file = os.path.join(img_folder, f'{title}.png')
    plt.savefig(output_file)
    plt.show()



def stacked_bar(df, title, x_var, y_var, label_series, label_size=15, y_lim=None, x_lim=None, text_condition=None,
                invert_xaxis=False, log_transformed=False, color_palette=None):
    sns.set_style("darkgrid")
    fig, ax1 = plt.subplots(figsize=(15, 12))
    # n_series = df[hue_var].nunique()
    if color_palette is None:
        color_palette = {k: i for k, i in
                         zip(label_series, sns.color_palette("Spectral", as_cmap=False, n_colors=len(label_series)))}
    df.plot(kind='bar', stacked=True, color=color_palette, ax=ax1)
    for c in ax1.containers[:-1]:
        labels = ['{:.1f}'.format(v.get_height()) if v.get_height() > 0 else '' for v in c]
        # remove the labels parameter if it's not needed for customized labels
        ax1.bar_label(c, labels=labels, label_type='center', fmt='.1f', padding=5, fontsize=label_size)

    # last_c = ax1.containers[0]

    for point in ax1.containers[-1]:
        val = point.get_height()
        label = '{:.1f}'.format(val) if val > 0 else ''
        ax1.text(point.xy[0] + point.get_width() * 0.36, point.xy[1] + val + 2.5, label, fontsize=label_size)

    ax1.tick_params(axis='both', labelsize=25)
    ax1.figure.autofmt_xdate(rotation=270, ha='center')
    plt.ylabel(y_var.replace('_', ' ').title(), size=30)
    plt.xlabel(x_var.replace('_', ' ').title(), size=30)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.02, 1.0), loc='upper left')
    plt.title(title, fontsize=35, y=1.03)
    # ax1.legend(bbox_to_anchor=(1.5, 1))

    if invert_xaxis:
        ax1.invert_xaxis()
    if log_transformed:
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    output_file = os.path.join(img_folder, f'{title}.png')
    plt.tight_layout()
    fig.savefig(output_file)

    plt.show()


def heatmap_df(df):
    sns.set_style("darkgrid")
    head_map_age = df.groupby(['first_year', 'age']).login.nunique().reset_index()
    head_map_age = pd.pivot_table(head_map_age, index=['first_year'], columns=['age'], values=['login'])
    head_map_age.columns = [i[1] for i in head_map_age.columns]
    head_map_age.columns.name = 'age'
    head_map_age_pct = head_map_age.copy()
    head_map_age_pct = head_map_age_pct.transpose()
    head_map_age_pct = head_map_age_pct.div(head_map_age_pct.iloc[0]) * 100
    head_map_age_pct = head_map_age_pct.transpose()
    heatmap(head_map_age_pct, 'Organization Cohort', 'Age Groups', 'Retention % of Organizations')

    tab_name = 'repo age retention'
    gs_reader.del_worksheet(gs_id, [tab_name])
    gs_reader.write_into_worksheet(head_map_age_pct.iloc[1:].mean().reset_index().fillna(''), gs_id, tab_name)


def get_local_data(df_created):
    folder = 'adhoc_data/github_biquery'
    # ['org_contributor_cnt.csv', 'org_repo_count.csv']
    dev = pd.read_csv(os.path.join(folder, 'org_contributor_cnt.csv'))
    dev = prepare_cdf(dev, 'contributors_count', 'year')

    dev['first_year'] = dev.groupby('login').year.transform('min')
    dev['age'] = dev['year'].astype(int) - dev['first_year'].astype(int)

    color_palette_ = sns.color_palette("Spectral", as_cmap=False, n_colors=8)
    color_palette = {k: v for k, v in zip(dev.year.unique(), color_palette_)}

    title = 'Distribution of # Contributors of Organizations'
    y_var = 'cumulative_pct'
    text_condition = f'rank_==1'
    single_axis_chart(dev.sort_values(by=['year', 'percentage'], ascending=[True, False]), title,
                      x_var='contributors_count', y_var='cumulative_pct', hue_var='year',
                      color_palette=color_palette, y_lim=[0, 100], text_condition=text_condition, invert_xaxis=True)

    title = 'CDF of # Contributors of Organizations'
    y_var = 'cdf'
    text_condition = 'cdf==1'
    single_axis_chart(dev, title, x_var='contributors_count', y_var=y_var, hue_var='year',
                      color_palette=color_palette, y_lim=[0.95, 1], text_condition=text_condition)

    dev_age = prepare_cdf_per_age_group(dev, 'contributors_count', 'year', 'age')
    title = 'CDF of # Contributors of Organizations per Age Group'
    y_var = 'cdf'
    text_condition = 'cdf==1'
    # group_key, fig_title, df_, x_var, y_var, hue_var
    multiple_subplot('age', title, dev_age, x_var='contributors_count', y_var=y_var, hue_var='year',
                     color_palette=color_palette, y_lim=[0.95, 1], text_condition=text_condition)

    title = 'Breakdown of Organizations by Age Groups'
    color_palette_age = {k: v for k, v in zip(list(range(8)), color_palette_)}
    age_cnt = dev[['year', 'age', 'login']].groupby(['year', 'age']).login.nunique().reset_index()
    age_cnt = age_cnt.rename(columns={'login': 'orgs_cnt'})
    single_axis_chart(age_cnt.query('year>15'), title, x_var='year', y_var='orgs_cnt', hue_var='age',
                      color_palette=color_palette_age, text_condition='year==22')

    title = '# Organizations by Age Groups'
    color_palette_age = {k: v for k, v in zip(list(range(8)), color_palette_)}
    age_cnt = dev[['year', 'age', 'login']].groupby(['year', 'age']).login.nunique().reset_index()
    age_cnt = age_cnt.rename(columns={'login': '#orgs'})
    single_axis_chart(age_cnt.query('year-15>age'), title, x_var='year', y_var='#orgs', hue_var='age',
                      color_palette=color_palette_age, text_condition='year==22')

    head_map_age = dev.groupby(['first_year', 'age']).login.nunique().reset_index()
    head_map_age = pd.pivot_table(head_map_age, index=['first_year'], columns=['age'], values=['login'])
    head_map_age.columns = [i[1] for i in head_map_age.columns]
    head_map_age.columns = 'age'
    head_map_age_pct = head_map_age.copy()
    head_map_age_pct = head_map_age_pct.transpose()
    head_map_age_pct = head_map_age_pct.div(head_map_age_pct.iloc[0]) * 100
    head_map_age_pct = head_map_age_pct.transpose()
    heatmap(head_map_age_pct, 'Organization Cohort', 'Age Groups', 'Retention % of Organizations')
    d = pd.DataFrame(head_map_age_pct.iloc[1:].mean()).reset_index().fillna('')
    gs_reader.write_into_worksheet(d, gs_id, 'retention')

    new_org_growth = (head_map_age[0].iloc[1:].pct_change(1) * 100).reset_index().fillna('')
    gs_reader.write_into_worksheet(new_org_growth, gs_id, 'new_org_growth')

    repo = pd.read_csv(os.path.join(folder, 'org_repo_count.csv'))
    repo = prepare_cdf(repo, 'repos_count', 'year')

    repo['first_year'] = repo.groupby('login').year.transform('min')
    repo['age'] = repo['year'].astype(int) - repo['first_year'].astype(int)
    repo_age = prepare_cdf_per_age_group(repo, 'repos_count', 'year', 'age')
    title = 'CDF of # Repo of Organizations per Age Group'
    y_var = 'cdf'
    text_condition = 'cdf==1'
    # group_key, fig_title, df_, x_var, y_var, hue_var
    multiple_subplot('age', title, repo_age, x_var='repos_count', y_var=y_var, hue_var='year',
                     color_palette=color_palette, y_lim=[0.95, 1], text_condition=text_condition)

    title = 'CDF of # Repos of Organizations'
    y_var = 'cdf'
    text_condition = 'cdf==1'
    single_axis_chart(repo, title, x_var='repos_count', y_var=y_var, hue_var='year',
                      color_palette=color_palette, y_lim=[0.95, 1], text_condition=text_condition)

    # only when the prepare_cdf was in descending order

    title = 'Distribution of # Repos within Organizations'
    y_var = 'cumulative_pct'
    text_condition = f'rank_==1'
    single_axis_chart(repo.sort_values(by=['year', 'percentage'], ascending=[True, False]), title,
                      x_var='repos_count', y_var='cumulative_pct', hue_var='year',
                      color_palette=color_palette, y_lim=[0, 100], text_condition=text_condition, invert_xaxis=True)

    df_created['first_date'] = pd.to_datetime(df_created['first_event_date']).map(lambda i: i.year)
    # df_created = repo.groupby('login').agg(first_date=('year', 'min')).reset_index()

    dev_df = dev.merge(df_created[['login', 'first_date']], on='login', how='left')
    dev_df['age'] = dev_df['year'].astype(int) - dev_df['first_date']

    repo_summary = repo.groupby('year').repos_count.describe(percentiles=[.5, .75, .80, .85, .90, .95, .98, .99])
    gs_reader.write_into_worksheet(repo_summary.reset_index(), gs_id, 'repo_sum')

    return


# plot 1
# repo - release
# organization bar
def multiple_subplot(group_key, fig_title, df_, x_var, y_var, hue_var, color_palette=None, y_lim=None,
                     text_condition=None, invert_xaxis=False):
    df_groups = [(k, v) for k, v in df_.groupby(group_key)]
    n_series = df_[hue_var].nunique()
    if color_palette is None:
        # color_palette = [*sns.color_palette("ch:s=.5,rot=-.25", as_cmap=False, n_colors=n_series)]
        # color_palette = sns.cubehelix_palette(start=2, n_colors=n_series)
        color_palette = {k: v for k, v in
                         zip(df_[hue_var].unique(), sns.color_palette("Spectral", as_cmap=False, n_colors=n_series))}
    fig, axes = plt.subplots(2, 4, figsize=(30, 20))
    for (title, df), ax1 in zip(df_groups, axes.flatten()):

        # sns.color_palette("Spectral", as_cmap=True)
        # sns.color_palette("Spectral", as_cmap=True)
        markers = ["o"] * n_series
        sns.lineplot(data=df, x=x_var, y=y_var,
                     hue=hue_var, ax=ax1, palette=color_palette, linewidth=2.5, markers=markers)
        sns.scatterplot(x=x_var, y=y_var, hue=hue_var, size=x_var, data=df, ax=ax1, palette=color_palette, s=10,
                        legend=False)

        # plt.legend(fontsize=18, loc='best')
        if y_lim is not None:
            ax1.set_ylim(y_lim)
        if text_condition is not None:
            for i, point in df.query(text_condition).iterrows():
                ax1.text(point[x_var], point[y_var], str(point[hue_var]), fontsize=18, color='grey', )
        # df.resample('M').sum().plot(ax=ax1)
        # df.resample('M').nunique().plot(ax=ax1)
        ax1.tick_params(axis='both', labelsize=15)
        ax1.legend(loc='lower right', fontsize=25)
        # plt.ylabel(y_var.replace('_', ' ').title(), size=15)
        # plt.xlabel(x_var.replace('_', ' ').title(), size=15)
        ax1.set_xlabel(x_var.replace('_', ' ').title(), size=15)
        ax1.set_ylabel(y_var.replace('_', ' ').title(), size=15)
        ax1.set_title(title, fontsize=20, y=1.03)
        if invert_xaxis:
            ax1.invert_xaxis()
        # plt.tight_layout()
    output_file = os.path.join(img_folder, f'{title}.png')
    fig.savefig(output_file)
    plt.show()


def double_axis_bar_line_plot(df_bar, df_line, x_var, y_bar, y_line, title, y_lim=None):
    # fig, ax1 = plt.subplots(figsize=(30, 10))
    fig, ax1 = plt.subplots(figsize=(30, 10))
    # plot line chart on axis #1
    ax1.bar(df_bar[x_var], df_bar[y_bar], width=0.5, alpha=0.5, color='orange')
    # ax1.set_ylim(0, 25)
    # ax1.legend(['average_temp'], loc="upper left")
    # set up the 2nd axis
    ax2 = ax1.twinx()
    # plot bar chart on axis #2
    ax2.plot(df_line[x_var], df_line[y_line])
    ax2.set_ylabel(y_line)

    ax2.grid(False)  # turn off grid #2
    plt.title(title, fontdict={'fontsize': 30})
    ax1.bar_label(ax1.containers[0], label_type='center', fontsize=18)

    ax1.yaxis.set_tick_params(labelsize=22)
    ax1.xaxis.set_tick_params(labelsize=22)
    ax2.yaxis.set_tick_params(labelsize=22)

    ax1.set_ylabel(y_bar.replace('_', ' ').title(), fontsize=25)
    ax2.set_ylabel(y_line.replace('_', ' ').title(), fontsize=25)
    for i, point in df_line.iterrows():
        ax2.text(point[x_var], point[y_line] * 0.95, '{:.1f}'.format(point[y_line]), fontsize=18, color='grey')
    if y_lim is not None:
        ax2.set_ylim(y_lim)
    plt.tight_layout()
    plt.show()
    output_file = os.path.join(img_folder, f'{title}.png')
    fig.savefig(output_file)


def single_axis_chart(df, title, x_var, y_var, hue_var, color_palette=None, y_lim=None, x_lim=None, text_condition=None,
                      invert_xaxis=False, log_transformed=False, figsize=None):
    sns.set_style("darkgrid")
    if figsize is None:
        figsize = (15, 12)
    fig, ax1 = plt.subplots(figsize=figsize)
    n_series = df[hue_var].nunique()
    if color_palette is None:
        # color_palette = [*sns.color_palette("ch:s=.5,rot=-.25", as_cmap=False, n_colors=n_series)]
        # color_palette = sns.cubehelix_palette(start=2, n_colors=n_series)
        color_palette = sns.color_palette("Spectral", as_cmap=False, n_colors=n_series)
    # sns.color_palette("Spectral", as_cmap=True)
    # sns.color_palette("Spectral", as_cmap=True)
    markers = ["o"] * n_series
    sns.lineplot(data=df, x=x_var, y=y_var,
                 hue=hue_var, ax=ax1, palette=color_palette, linewidth=2.5, markers=markers)
    sns.scatterplot(x=x_var, y=y_var, hue=hue_var, size=x_var, data=df, ax=ax1, palette=color_palette, s=10,
                    legend=False)

    plt.legend(fontsize=18, loc='best')
    if y_lim is not None:
        ax1.set_ylim(y_lim)
    if x_lim is not None:
        ax1.set_xlim(x_lim)
    if text_condition is not None:
        for i, point in df.query(text_condition).iterrows():
            ax1.text(point[x_var], point[y_var], str(point[hue_var]), fontsize=18, color='grey', )
    # df.resample('M').sum().plot(ax=ax1)
    # df.resample('M').nunique().plot(ax=ax1)
    ax1.tick_params(axis='both', labelsize=25)
    plt.ylabel(y_var.replace('_', ' ').title(), size=30)
    plt.xlabel(x_var.replace('_', ' ').title(), size=30)
    plt.title(title, fontsize=35, y=1.03)
    if invert_xaxis:
        ax1.invert_xaxis()
    if log_transformed:
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    output_file = os.path.join(img_folder, f'{title}.png')
    fig.savefig(output_file)
    plt.show()


def heatmap(head_map_df, y_var, x_var, title):
    fig, ax1 = plt.subplots(figsize=(15, 12))
    # sns.set(font_scale=1.4)
    sns.heatmap(head_map_df,
                cmap='crest',
                annot=True,
                annot_kws={'fontsize': 18},
                fmt='.1f',
                vmax=head_map_df[0].max(),
                ax=ax1,
                )
    plt.ylabel(y_var.replace('_', ' ').title(), size=30)
    plt.xlabel(x_var.replace('_', ' ').title(), size=30)
    ax1.tick_params(axis='both', labelsize=25)
    plt.title(title, fontsize=35, y=1.03)
    output_file = os.path.join(img_folder, f'{title}.png')
    fig.savefig(output_file)
    plt.show()
    return


if __name__ == '__main__':
    pass
