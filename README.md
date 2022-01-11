# visualization-politic-regimes


## Introduction
Does democracy lead to good governance outcomes? Assessing the role that political regimes play in determining economic and human development outcomes is complicated, not least because it requires disentangling the effect of historical contingencies from that of the regime itself. The literature on the connection between democracy and governance outcomes is somewhat divided. While conventional wisdom has it that democracy promotes good human development outcomes, the same does not hold true for economic outcomes—scholars have generally come to the conclusion that the causal relationship between democracy and economic growth is complex and difficult to characterize , if at all it exists (Przeworski and Limongi, 1993; Glaeser et al., 2004; Acemoglu e al., 2008). Due to time constraints, this project will focus only on an initial investigation of the relationship between political regimes and governance outcomes. We first retrieved the Democracy Index scores, as well as economic and human development data through web scraping and PDF parsing . We then created a series of static and interactive plots to illustrate the relationship between political regimes and governance outcomes; lastly, we fit a simple OLS model to the data. The limitations of our approach are, of course, many and varied, as are the potential avenues for future research, and will be discussed in brief below. 

## Results
The Economist Intelligence Unit’s Democracy Index score is based on five different categories: electoral process and pluralism; the functioning of government; political participation; political culture; and civil liberties. Based on its overall score, each country is then classified as one of four kinds of political regimes: “full democracy,” “flawed democracy,” “hybrid regime,” or “authoritarian regime.” Data are available for the years 2006, 2008, and 2010 – 2020. We have accordingly restricted our final dataset to the years 2010 – 2020. In the interests of space, only the more salient plots will be discussed below. 

![KDE for economic indicators](https://user-images.githubusercontent.com/81766373/149034481-bf42fea6-150f-45cd-8ab0-41d2d9ac12e0.png)

![KDE for human development indicators](https://user-images.githubusercontent.com/81766373/149034565-f929fe3c-ce36-4ed1-9c8e-1963e1d16112.png)

To simplify the analysis, we have compared only full democracies and authoritarian regimes . From the kernel density (KDE) plots above, it is easy to see that there are fairly strong relationships between types of political regimes and economic outcomes, and that full democracies consistently outperform authoritarian regimes on all four of the economic indicators investigated. Full democracies have a much higher GDP per capita (PPP) on average than authoritarian regimes do, despite the existence of some outliers. (Perhaps unsurprisingly, the most extreme outlier of the authoritarian regimes is Qatar.) Authoritarian regimes tend to have fatter tails for unemployment rates and income inequality (the Gini coefficient) than full democracies do, whereas the opposite is true for foreign investment.

![GDP by type of political regime](https://user-images.githubusercontent.com/81766373/149034711-d59e7e94-12cb-478f-bcda-b9ba5f8371fa.png)

Likewise, there are strong relationships between political regimes and human development outcomes: On nearly every metric except for CO2 emissions and forest area (in square kilometers), full democracies consistently outperform authoritarian regimes. It is worth noting that full democracies in fact perform worse than authoritarian regimes do where CO2 emissions are concerned—which is, again, unsurprising, in light of the fact that developed countries (which also tend to be full democracies) have historically been responsible for producing the most carbon emissions (both in aggregate, and per capita). If the density plot below is any indication, however, it seems to suggest that authoritarian regimes and/or developing countries are rather rapidly catching up—which is far from good news.

![DI score by sub-region](https://user-images.githubusercontent.com/81766373/149034870-3e7380ac-6507-4c00-8ed0-039866c8a77e.png)

## Interactive plots
Of especial interest is the first scatterplot (the one shown below is for the year 2019, since the most recent year for which HDI data are available is 2019): Across the years 2010 – 2020, there is a strong positive correlation between the Human Development Index (HDI) and GDP per capita (PPP), as well as between each metric and the Democracy Index score. This accords with the evidence shown above; but as will be discussed in the section below, it is only a fairly preliminary investigation of the relationship between political regimes and governance outcomes, and should be taken with a grain of salt.

![interactive_plot](https://user-images.githubusercontent.com/81766373/149035002-872e3fdd-fa9d-427a-bae3-f113efcecf26.png)

