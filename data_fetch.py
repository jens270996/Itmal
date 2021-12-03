from types import TracebackType
import spotipy
import csv
import math

classical_uris=["7v9uKvGwd9pwloVMziYPAg","6pYj1EckIAU4TC068BBIz6","3BAtH3eZHdi4vfN6ZkpGiX","37i9dQZF1EId314iwsa9iP","37i9dQZF1DWWEJlAGA9gs0", 
    "1h0CEZCm6IbFTbxThn6Xcs", "37i9dQZF1EIeLlBsaQNENK", "3ttT3EIioY5KlSBI3l4u5H", "5A0hTaFKvOcGZPIQ3zr6aT", "37i9dQZF1DX3PFzdbtx1Us", 
    "3Zh1DAcvLngbd1qiBYEex0", "1Z7fO3bkVteGsTbVluOQoH", "36qQ3naFyPxCbbr7QpdADN", "1HY6rzP7xwiIolIXRsKAbW", "4LXbPDPBy6FLZmtYa3ooim", 
    "7LlzRvcCHFjWHA89IMDbDV", "1t1mH9Xh2v3aH6yJfMlXeP", "32UsbyNeugCxRo6SLvzZ6Y", "48pLGJFuMk12HgVWDdgVA5"]
jazz_uris=["37i9dQZF1DXbITWG1ZJKYt", "37i9dQZF1DX3SiCzCxMDOH", "37i9dQZF1DXb1JVnfCng2p", "40VS343V2hVFe3GGP55ZWQ", "37i9dQZF1EId4oprFkTPb8", 
    "5pcU1JB2yM2f5OO90PQyAO", "37i9dQZF1DX2iVTU1bf67i", "6GNoAopv64BlP5oFXrGbza", "37i9dQZF1EIehTvTm4HAQL", "37i9dQZF1EIgAElrSienLL", 
    "37i9dQZF1EIf1zoXgYNkes", "37i9dQZF1EIcNF0RmPTgbB", "00EDJ9x7DEBpw1Eva8s9US", "37i9dQZF1EIf5KE70N4BpI", "0sJugi8ocE7d7nsO4r2Gm4", 
    "2AQvOtggQ2mHFdCeWnjcLY", "37i9dQZF1DX55dNU0PWnO5"]
rock_uris=["37i9dQZF1EIdFxMZckHt54","6mFshQVVjkGRDhvZyXS2gM","42EEAVMDLQmpwACJ3VlhKW","37i9dQZF1DWXRqgorJj26U", "37i9dQZF1DWZNFWEuVbQpD", 
    "34NjkEHmcgjPPaw9uUtpEp", "0qkUCHf4mOb6JutykbutB0", "0cCG38OjT3qO0Pd4mg8j1z", "5uEuSIS36zZtsne5go7PhI", "0ixebA2tXOSkpV8xab7q1B", 
    "29fOXPbR3qClySksICQ2Yn"]
techno_uris=["1qM9FkyCiljivkQvkV7DKJ","5lSgExb6yTKfLusGag7bm7","0HPNBM0iY3sYGEY2zftqlO","3ifXguT2UIneWLvg0jlZc3","6Q4pnSDA3RtYMB0AVNRv65",
    "7mwPa6HjqoiUrsk3C2Hitk", "37i9dQZF1DX6J5NfMJS675","37i9dQZF1EIeKh45OZ1ylm", "6jCgabXrYUjqdaF0ozKkTj", "0itqm5PKjtaSuKaVNQ5KWb", 
    "18vUeZ9BdtMRNV6gI8RnR6", "4nEuCsvecNXuQ4B6lCOlFl", "2pX5seO5zvYeGEA9HX91yE", "561d05MakkMq23apDVqOvd", "0BcS6dygtX4L3MHPEP0F2k", 
    "37i9dQZF1DX2R0a3scWaq6", "1R6agaYjDLeo779oDZKCQ0"]


playlist_uris=[classical_uris,jazz_uris,rock_uris,techno_uris]
playlist_genres=[0,1,2,3]
playlists=[]
track_objects=[]
song_count=0
# https://developer.spotify.com/documentation/web-playback-sdk/quick-start/#

spotify = spotipy.Spotify(client_credentials_manager=spotipy.oauth2.SpotifyClientCredentials(
    client_id="b0cfe3cff3e04e1c83869890f473549e",client_secret="d89a0d6c74084454b1b82d2ee6b74e07"))
# spotify.current_user()
#spotify.playlist(playlist_id=playlist)["tracks"]["total"]
for playlist_list,genre in zip(playlist_uris,playlist_genres):
    for playlist in playlist_list:
        palyistlist_dict= spotify.playlist(playlist_id=playlist)
        playlists.append((palyistlist_dict["tracks"]["items"],genre))
        total = palyistlist_dict["tracks"]["total"]
        print("total is:",total)
        for i in range(1,math.ceil((total)/100)):
            print("hi from loop it:",i)
            playlists.append((spotify.playlist_items(playlist_id=playlist,offset=i*100)["items"],genre))


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
        track_object.pop('type')
        track_objects.append(track_object)
    print("Track_objects:",len(track_objects))
    print("Song count:",song_count)

#cross iterate track_objects, and remove duplicates.
print("Before removing duplicates:",len(track_objects))
track_objects=list({v['id']: v for v in track_objects}.values())
print("After removing duplicates:",len(track_objects))

for track in track_objects:
    track.pop('id')
print(dict.keys(track_objects[0]))
fieldnames=dict.keys(track_objects[0])
with open('spotify_data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(track_objects)
