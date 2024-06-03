import React from 'react';
import { Card, Text } from 'react-native-paper';
import { View, ImageBackground, StyleSheet, ScrollView } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { TouchableOpacity } from 'react-native-gesture-handler';



function Courasel() {
    const cardData = [
        {
          id: 1,
          category: '5 Warm-ups for the Upper-body',
          imageUrl: require('../assets/images/Frame 1.png'),
          time: '15 minutes | 3 sets',
          videoUrl: require('../assets/videos/superman.gif'),
        },
        {
          id: 2,
          category: 'Sit-ups in 10 minutes',
          imageUrl: require('../assets/images/Frame 2.png'),
          time: '10 minutes | 3 sets',
          videoUrl: require('../assets/videos/legraises.gif'), // Add video URL
        },
        {
          id: 3,
          category: 'Pushup for Beginnes',
          imageUrl: require('../assets/images/Frame 3.png'),
          time: '15 minutes | 3 sets',
          videoUrl: require('../assets/videos/pushups.gif'), // Add video URL
        },
      ];

      const cardData2 = [
        {
          id: 1,
          category: 'Plank with Shoulder Taps',
          imageUrl: require('../assets/images/Frame 2_1.png'),
          time: '6 minutes | 3 sets',
          videoUrl: require('../assets/videos/pushups.gif'), // Add video URL
        },
        {
          id: 2,
          category: 'Mountain Climber for Beginners',
          imageUrl: require('../assets/images/Frame 2_2.png'),
          time: '15 minutes | 3 sets',
          videoUrl: require('../assets/videos/curl.gif'), // Add video URL
        },
        {
          id: 3,
          category: 'Lunges with Light Weights',
          imageUrl: require('../assets/images/Frame 2_3.png'),
          time: '15 minutes | 3 sets',
          videoUrl: require('../assets/videos/squats.gif'), // Add video URL
        },
      ];

  const navigation = useNavigation();

  const handleCardPress = (videoUrl) => {
    navigation.navigate('VideoScreen', { videoUrl });
  };

  return (
    <SafeAreaProvider>
      <ScrollView>
        <View style={styles.featuredContainer}>
          <Text style={styles.headingText}>Upper Body</Text>
          <ScrollView contentContainerStyle={styles.cardScrollView} horizontal={true} showsHorizontalScrollIndicator={false} pagingEnabled={false}>
            {cardData.map((card) => (
              <TouchableOpacity key={card.id} onPress={() => handleCardPress(card.videoUrl)}>
                <Card style={styles.cardContainer}>
                  <ImageBackground style={styles.cardImage} resizeMode='cover' source={card.imageUrl}>
                    <Text style={styles.cardTitle}>{card.category}</Text>
                    <Text style={styles.cardTime}>{card.time}</Text>
                  </ImageBackground>
                </Card>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
        <View style={styles.featuredContainer}>
          <Text style={styles.headingText}>Core</Text>
          <ScrollView contentContainerStyle={styles.cardScrollView} horizontal={true} showsHorizontalScrollIndicator={false} pagingEnabled={false}>
            {cardData2.map((card) => (
              <TouchableOpacity key={card.id} onPress={() => handleCardPress(card.videoUrl)}>
                <Card style={styles.cardContainer}>
                  <ImageBackground style={styles.cardImage} resizeMode='cover' source={card.imageUrl}>
                    <Text style={styles.cardTitle}>{card.category}</Text>
                    <Text style={styles.cardTime}>{card.time}</Text>
                  </ImageBackground>
                </Card>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      </ScrollView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  },
  searchContainer: {
    width: '100%',
    height: 60,
    marginTop: 10
  },
  searchBar: {
    backgroundColor: 'white'
  },
  cardScrollView: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 5
  },
  featuredContainer: {
    width: '100%',
    height: 280,
    marginTop: 20,
    display: 'flex',
  },
  industriesContainer: {
    width: '100%',
    height: 500,
    marginTop: 15
  },
  funContainer: {
    width: '100%',
    height: 350,
    marginTop: 15
  },
  cardContainer: {
    width: 220,
    height: 240,
    marginLeft: 15,
    marginTop: 15
  },
  cardTitle: {
    width: 200,
    color: 'white',
    textTransform: 'uppercase',
    fontSize: 20,
    fontWeight: '800',
    marginTop: 150,
    marginLeft: 10
  },
  cardTime: {
    color: 'white',
    fontSize: 15,
    fontWeight: '600',
    marginRight: 60
  },
  cardImage: {
    alignItems: 'center',
    width: 220,
    height: 240,
    borderRadius: 10,
    overflow: 'hidden'
  },
  industriesCategories: {
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
    width: '100%',
    height: 60,
    marginTop: 15
  },
  categoriesImage: {
    width: '15%',
    height: 50,
    marginLeft: 25,
    borderRadius: 15
  },
  categoriesText: {
    marginLeft: 50,
    fontWeight: '700',
    fontSize: 20
  },
  headingText: {
    marginLeft: 35,
    fontWeight: '900',
    fontSize: 15,
    color: 'black'
  },
  image: {
    width: '100%',
    height: 50,
    borderRadius: 15,
    overflow: 'hidden'
  },
  addCategroy: {
    width: 50,
    height: 20,
    fontWeight: '700',
    fontSize: 20,
    position: 'absolute',
    right: 20
  }
});

export default Courasel;
