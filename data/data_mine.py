import tensorflow as tf 
import json 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 

def load_data():
    # try:
    with open('total_rewards.json','r', encoding='utf-8') as data_file:
        data = json.load(data_file)
    print("Data loaded")
    # except:
    #     data = {}
    #     print("Error loading data from 'total_rewards.json'.")
    return data


def show_reward_graph(data):
    num_of_reward = len(data['rewards_per_episode'])
    temp = {'episode':[],'reward':[]}
    for episode in range(num_of_reward):
        if (data['rewards_per_episode'][episode]>0):
            temp['episode'].append(episode)
            temp['reward'].append(data['rewards_per_episode'][episode])
    df = pd.DataFrame(temp, columns =['episode','reward'])
    df.plot(kind='line',x='episode',y='reward',color='red')
    plt.savefig('reward_visualization.png')
    plt.show()

def loss_distribution_graph(data):
    q_data = data['q_values']
    df = pd.DataFrame(q_data)
    old_q = (df['old_q'])
    new_q = df['new_q']
    prediction = tf.placeholder("float",[None])
    optimal = tf.placeholder("float",[None])
    loss = tf.losses.mean_squared_error(prediction,optimal)
    
    temp = {'episode':[], 'loss':[]}

    with tf.Session() as sess:
        for episode in range(len(df)):
            x = sess.run(loss, feed_dict = {prediction: old_q[episode], optimal:new_q[episode]})
            current_loss = x/len(new_q[episode])
            if (current_loss>0):
                temp['episode'].append(episode)
                temp['loss'].append(current_loss)
    df = pd.DataFrame(temp)
    df['loss'] = (df['loss'] - df['loss'].mean())/ (df['loss'].max() - df['loss'].min())
    df.plot(kind='line',x='episode',y='loss',color='red')
    plt.savefig('loss_visualization.png')
    plt.show()

if __name__ == "__main__":
    data = load_data()
    show_reward_graph(data)
    loss_distribution_graph(data)
