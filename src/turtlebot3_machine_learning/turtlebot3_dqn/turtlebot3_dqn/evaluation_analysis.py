import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_w1=pd.read_csv('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/Evaluation_RL_Klassisch_Log/Stage1_AS5_RewardPaper_AlleKomponenten.csv',sep=';')
df_w2=pd.read_csv('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/Evaluation_RL_Klassisch_Log/Stage2_AS5_RewardPaper_AlleKomponenten.csv',sep=';')
#df_w3=pd.read_csv('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/Evaluation_RL_Klassisch_Log/Stage3_Agent1.csv',sep=';')
#df_w4=pd.read_csv('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/Evaluation_RL_Klassisch_Log/Stage4_Agent1.csv',sep=';')
#df_w5=pd.read_csv('/home/verwalter/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/Evaluation_RL_Klassisch_Log/Stage5_Agent1.csv',sep=';')
print(len(df_w1))


def analyze_metrics(data:pd.DataFrame):
    print(len(data))
    data_sucess = data[data['success/collision'] == 'success']
    sucess_rate = len(data_sucess)
    collision_rate  = len(data) - len(data_sucess)
    spl = data_sucess['path_efficiency'].mean()
    sct = data_sucess['time_effiency'].mean()
    return sucess_rate,collision_rate,spl,sct
    
a,b,c,d = analyze_metrics(df_w1)
print(f'sucess:{a}, collision:{b}, SPL:{c}, SCT:{d}')
a,b,c,d = analyze_metrics(df_w2)
print(f'sucess:{a}, collision:{b}, SPL:{c}, SCT:{d}')

print(df_w1['success/collision'].value_counts())

def plot_data(w1,w2):#,w2,w3,w4,w5):
    list = [w1,w2]#,w2,w3,w4,w5]
    df_all = pd.DataFrame()
    for id,df in enumerate(list,1):
        df_all[f'Welt {id}'] = df['success/collision'].value_counts()
    return df_all
    

    
plot_df = plot_data(df_w1,df_w2)#,df_w2,df_w3,df_w4,df_w5)
plot_df2 = plot_df.copy()
plot_df2.index.name = "Ergebnis"          # <- Name für den Index setzen
plot_df_long = (plot_df2.reset_index().melt(id_vars="Ergebnis", var_name="Welt", value_name="Anzahl"))
print(plot_df_long)



plot_df_long.rename(columns={'index': 'Ergebnis'}, inplace=True)


sns.set_theme()
fig = plt.figure(figsize=(10, 8))
fig.set_constrained_layout(True)
gs = fig.add_gridspec(2, 2)
plot1 = fig.add_subplot(gs[0, 0]);plot1.grid(True)
plot2 = fig.add_subplot(gs[0, 1]);plot2.grid(True)
plot3 = fig.add_subplot(gs[1, :]);plot3.grid(True)

plot3 = sns.barplot(data=plot_df_long,x="Welt",y="Anzahl",hue="Ergebnis")


plt.show()

