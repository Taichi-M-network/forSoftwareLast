import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def getTrackIDs(user, playlist_id):
    ids = []
    # playlist = sp.user_playlist(user, playlist_id)
    playlist = sp.playlist(playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        ids.append(track['id'])

    return ids


def getTrackFeatures(id):
    meta = sp.track(id)
    features = sp.audio_features(id)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    mode = features[0]['mode']
    liveness = features[0]['liveness']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    time_signature = features[0]['time_signature']
    valence = features[0]['valence']

    track = [name, album, artist, release_date, length,
             mode, popularity, danceability, acousticness, energy,
             instrumentalness, liveness, loudness, speechiness, tempo,
             time_signature, valence]
    return track


if __name__ == '__main__':
    # spotify developerから取得したclient_idとclient_secretを入力
    client_id = 'e642c012a3a343509862ea677a67abb3'
    client_secret = '4d432c78601b45d59bdf963907c90cd6'

    client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Spotifyのユーザー名と、プレイリストのIDを入力
    username = "uvvis7n6ldvp71s3dg55zgr55"
    playlist_id = "5dsTEO8o47WNdllWWeYhsr"
    # getTrackIDs呼び出し
    ids = getTrackIDs(username, playlist_id)

    # loop over track ids
    tracks = []
    for i in range(len(ids)):
        time.sleep(.5)
        track = getTrackFeatures(ids[i])
        tracks.append(track)

    # create dataset
    df = pd.DataFrame(tracks, columns=['name', 'album', 'artist', 'release_date', 'length',
                                       'mode', 'popularity', 'danceability', 'acousticness', 'energy',
                                       'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',
                                       'time_signature', 'valence'])
    """
    name:曲名、album:アルバム、artist:アーティスト名、release_date:リリース日、length:曲の長さ、
    popularity:人気度、danceability:ダンス度、acousticness:アコースティック度、energy:エネルギー、
    instrumentalness:インスト感、mode:曲調(メジャー:1/マイナー:0)、liveness:ライブさ、loudness:曲の大きさ、speechiness:スピーチ度、
    valance:ポジティブ度、tempo:テンポ、time_signature:拍子
    """

    # データ列絞り込み
    df_sub = df[["popularity", "energy", "mode", "tempo"]]

    # 標準化
    sc = StandardScaler()
    sc.fit(df_sub)
    df_sub_std = sc.transform(df_sub)

    # KMeansクラスの初期化
    kmeans = KMeans(init="random", n_clusters=6, random_state=0)

    # クラスターの重心の計算
    kmeans.fit(df_sub_std)

    # クラスター番号をpandasのSeriesオブジェクトに変換
    labels = pd.Series(kmeans.labels_, name="cluster_number")

    # クラスター番号と件数を表示
    # print(labels.value_counts(sort=False))

    # グラフの描画
    ax = labels.value_counts(sort=False).plot(kind="bar")
    ax.set_xlabel("cluster number")
    ax.set_ylabel("count")
    plt.show()