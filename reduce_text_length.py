from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
text = "Black widow spiders mate in spring and early summer.  The male black widow spider is about half the size of the female.  Though the female black widow does sometimes eat her mate after breeding, this is not always or even usually the case.  When the female black widow does feed on her mate it is because she is low in nutrients and needs the energy. After mating, the female black widow weaves a round or pear shaped nest of silk into which she deposits her eggs.  The web of the black widow is random and messy looking, not symmetrical and neat like some spiders.  Female black widows can mate and lay three or more egg sacs a season.  Hundreds of eggs are laid in each egg sac.  The mother black widow guards the egg sac for up to a month until it hatches. When the baby spiders hatch they begin to eat one another for nutrients, so only a few from each egg sac survive.  The baby survivors leave the egg sac within a couple of days of hatching.  When the baby spiders leave they often 'balloon'.  The baby spiders spin a long filament of web and it catches a breeze and floats them away from home. The baby spiders molt several times over the summer and fall.  Over winter the baby spiders mature. In the spring the mature spiders are ready to find their own mates and start the cycle over again.  Most black widows live only one year, but they can live up to three years."
tokenized_words = word_tokenize(text)

tokenized_without_stop_words = [word for word in tokenized_words if word not in stop_words]

separator = " "
text_without_stop_words = separator.join(tokenized_without_stop_words)
print(text_without_stop_words)
