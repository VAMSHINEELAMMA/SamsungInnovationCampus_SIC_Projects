"use client";

import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Brain, Code, FlaskConical, Globe, Palette, Mic, BookOpen, Atom, Award, Bot
} from "lucide-react";

const icons = [
  Brain, Code, FlaskConical, Globe, Palette, Mic, BookOpen, Atom
];
const allIcons = [...icons, ...icons];

type CardState = {
  icon: React.ElementType;
  isFlipped: boolean;
  isMatched: boolean;
};

const shuffleArray = (array: any[]) => {
  return array.sort(() => Math.random() - 0.5);
};

export function MemoryGame() {
  const [cards, setCards] = useState<CardState[]>([]);
  const [flippedIndices, setFlippedIndices] = useState<number[]>([]);
  const [moves, setMoves] = useState(0);

  const initializeGame = () => {
    const shuffledIcons = shuffleArray(allIcons);
    setCards(
      shuffledIcons.map((Icon, index) => ({
        icon: Icon,
        isFlipped: false,
        isMatched: false,
      }))
    );
    setFlippedIndices([]);
    setMoves(0);
  };

  useEffect(() => {
    initializeGame();
  }, []);

  useEffect(() => {
    if (flippedIndices.length === 2) {
      const [firstIndex, secondIndex] = flippedIndices;
      const firstCard = cards[firstIndex];
      const secondCard = cards[secondIndex];

      if (firstCard.icon === secondCard.icon) {
        // Match
        const newCards = [...cards];
        newCards[firstIndex].isMatched = true;
        newCards[secondIndex].isMatched = true;
        setCards(newCards);
        setFlippedIndices([]);
      } else {
        // No match
        setTimeout(() => {
          const newCards = [...cards];
          newCards[firstIndex].isFlipped = false;
          newCards[secondIndex].isFlipped = false;
          setCards(newCards);
          setFlippedIndices([]);
        }, 1000);
      }
      setMoves(moves + 1);
    }
  }, [flippedIndices, cards, moves]);

  const handleCardClick = (index: number) => {
    if (flippedIndices.length >= 2 || cards[index].isFlipped) {
      return;
    }
    const newCards = [...cards];
    newCards[index].isFlipped = true;
    setCards(newCards);
    setFlippedIndices([...flippedIndices, index]);
  };
  
  const isGameWon = cards.length > 0 && cards.every(c => c.isMatched);

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="grid grid-cols-4 gap-4">
        {cards.map((card, index) => (
          <Card
            key={index}
            onClick={() => handleCardClick(index)}
            className={cn(
              "h-20 w-20 md:h-24 md:w-24 flex items-center justify-center cursor-pointer transition-transform duration-500 transform-style-3d",
              card.isFlipped || card.isMatched ? "rotate-y-180" : "",
              card.isMatched ? "border-green-500" : ""
            )}
          >
            <div className="absolute backface-hidden">
                <div className="h-full w-full bg-secondary rounded-lg" />
            </div>
            <div className="rotate-y-180 backface-hidden">
                <card.icon className="h-10 w-10 text-primary" />
            </div>
          </Card>
        ))}
      </div>
      {isGameWon ? (
        <Card className="p-6 text-center bg-green-100 dark:bg-green-900/50 border-green-500">
          <Award className="h-12 w-12 text-green-500 mx-auto mb-4"/>
          <h3 className="text-xl font-bold">You Won!</h3>
          <p className="text-muted-foreground">You completed the game in {moves} moves.</p>
        </Card>
      ) : (
         <div className="text-lg font-semibold">Moves: {moves}</div>
      )}
      <Button onClick={initializeGame}>Reset Game</Button>
      <style jsx>{`
        .transform-style-3d { transform-style: preserve-3d; }
        .rotate-y-180 { transform: rotateY(180deg); }
        .backface-hidden { backface-visibility: hidden; }
      `}</style>
    </div>
  );
}
