
"use client";

import { useState, useEffect, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Brain, Code, FlaskConical, Globe, Palette, Mic, BookOpen, Atom, Award
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
  return array
    .map(value => ({ value, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ value }) => value);
};

export function MemoryGame() {
  const [cards, setCards] = useState<CardState[]>([]);
  const [flippedIndices, setFlippedIndices] = useState<number[]>([]);
  const [moves, setMoves] = useState(0);
  const [isChecking, setIsChecking] = useState(false);

  const initializeGame = useCallback(() => {
    const shuffledIcons = shuffleArray(allIcons);
    setCards(
      shuffledIcons.map((Icon) => ({
        icon: Icon,
        isFlipped: false,
        isMatched: false,
      }))
    );
    setFlippedIndices([]);
    setMoves(0);
    setIsChecking(false);
  }, []);

  useEffect(() => {
    initializeGame();
  }, [initializeGame]);

  useEffect(() => {
    if (flippedIndices.length === 2) {
      setIsChecking(true);
      const [firstIndex, secondIndex] = flippedIndices;
      const firstCard = cards[firstIndex];
      const secondCard = cards[secondIndex];

      if (firstCard.icon === secondCard.icon) {
        // Match
        setCards(prevCards => {
            const newCards = [...prevCards];
            newCards[firstIndex].isMatched = true;
            newCards[secondIndex].isMatched = true;
            return newCards;
        });
        setFlippedIndices([]);
        setIsChecking(false);
      } else {
        // No match
        setTimeout(() => {
          setCards(prevCards => {
            const newCards = [...prevCards];
            newCards[firstIndex].isFlipped = false;
            newCards[secondIndex].isFlipped = false;
            return newCards;
          });
          setFlippedIndices([]);
          setIsChecking(false);
        }, 1000);
      }
      setMoves(m => m + 1);
    }
  }, [flippedIndices, cards]);

  const handleCardClick = (index: number) => {
    if (isChecking || flippedIndices.length >= 2 || cards[index].isFlipped || cards[index].isMatched) {
      return;
    }
    
    setCards(prevCards => {
        const newCards = [...prevCards];
        newCards[index].isFlipped = true;
        return newCards;
    });

    setFlippedIndices(prev => [...prev, index]);
  };
  
  const isGameWon = cards.length > 0 && cards.every(c => c.isMatched);

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="grid grid-cols-4 gap-4">
        {cards.map((card, index) => {
          const Icon = card.icon;
          return (
            <Card
              key={index}
              onClick={() => handleCardClick(index)}
              className={cn(
                "h-20 w-20 md:h-24 md:w-24 flex items-center justify-center cursor-pointer transition-colors",
                card.isFlipped || card.isMatched ? "bg-card" : "bg-secondary",
                card.isMatched ? "border-green-500" : "border-border"
              )}
            >
              {(card.isFlipped || card.isMatched) && <Icon className="h-10 w-10 text-primary" />}
            </Card>
          );
        })}
      </div>
      {isGameWon ? (
        <Card className="p-6 text-center bg-green-100 dark:bg-green-900/50 border-green-500 animate-in fade-in zoom-in-95">
          <Award className="h-12 w-12 text-green-500 mx-auto mb-4"/>
          <h3 className="text-xl font-bold">You Won!</h3>
          <p className="text-muted-foreground">You completed the game in {moves} moves.</p>
        </Card>
      ) : (
         <div className="text-lg font-semibold">Moves: {moves}</div>
      )}
      <Button onClick={initializeGame}>Reset Game</Button>
    </div>
  );
}
