from types import TracebackType
import spotipy
import csv

playlist_uris=["37i9dQZF1DWWEJlAGA9gs0"
,"spotify:playlist:37i9dQZF1DXbITWG1ZJKYt","37i9dQZF1DWXRqgorJj26U","37i9dQZF1DX6J5NfMJS675"]
playlist_genres=[0,1,2,3]
playlists=[]
track_objects=[]
song_count=0
# https://developer.spotify.com/documentation/web-playback-sdk/quick-start/#

spotify = spotipy.Spotify(client_credentials_manager=spotipy.oauth2.SpotifyClientCredentials(
    client_id="b0cfe3cff3e04e1c83869890f473549e",client_secret="d89a0d6c74084454b1b82d2ee6b74e07"))
# spotify.current_user()
for playlist,genre in zip(playlist_uris,playlist_genres):    
    playlists.append((spotify.playlist(playlist_id=playlist)["tracks"]["items"]
    +spotify.playlist_items(playlist_id=playlist,offset=100)["items"],genre))

for tracks,genre in playlists:
    print("Running for genre:",genre)
    

    for track in tracks:
        song_count=song_count+1
        audio_features=spotify.audio_features(track["track"]["uri"])
        detailed_track=spotify.track(track["track"]["uri"])
        audio_feature=audio_features[0]
            #"Explicit":track["explicit"],"Duration":track["duration_ms"],"Popularity":track["popularity"]
        track_object={"genre":genre,"popularity":detailed_track["popularity"]}
        track_object.update(audio_feature)
        track_object.pop('analysis_url')
        track_object.pop('track_href')
        track_object.pop('uri')
        track_object.pop('id')
        track_object.pop('type')
        track_objects.append(track_object)
    print("Track_objects:",len(track_objects))
    print("Song count:",song_count)

print(dict.keys(track_objects[0]))
fieldnames=dict.keys(track_objects[0])
with open('spotify_data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(track_objects)
