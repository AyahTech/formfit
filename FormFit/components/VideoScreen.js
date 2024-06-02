import React from 'react';
import { View, StyleSheet, Text, Image } from 'react-native';
import { useRoute } from '@react-navigation/native';

const VideoScreen = () => {
  const route = useRoute();
  const { videoUrl } = route.params;

  return (
    <View style={styles.container}>
      {videoUrl ? (
        <Image
          source={videoUrl}
          style={styles.gif}
          resizeMode="contain"
        />
      ) : (
        <Text style={styles.errorText}>No GIF URL provided</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'black',
  },
  gif: {
    width: '100%',
    height: '100%',
  },
  errorText: {
    color: 'white',
    fontSize: 18,
  },
});

export default VideoScreen;
