use std::str::FromStr;

use crate::SnailfishNumber;
use itertools::Itertools;

#[test]
fn test_explode() {
    let input_expected = [
        ("[[[[[9,8],1],2],3],4]", "[[[[0,9],2],3],4]"),
        ("[7,[6,[5,[4,[3,2]]]]]", "[7,[6,[5,[7,0]]]]"),
        ("[[6,[5,[4,[3,2]]]],1]", "[[6,[5,[7,0]]],3]"),
        // multi-explode
        (
            "[[3,[2,[1,[7,3]]]],[6,[5,[4,[3,2]]]]]",
            "[[3,[2,[8,0]]],[9,[5,[7,0]]]]",
        ),
    ];
    for (input, expected) in input_expected {
        let mut num = SnailfishNumber::from_str(input).unwrap();
        num.explode_nums();
        let expected = SnailfishNumber::from_str(expected).unwrap();
        assert_eq!(num, expected);
    }
}

#[test]
fn test_add_snailfish_nums() {
    let num1 = SnailfishNumber::from_str("[[[0,[4,5]],[0,0]],[[[4,5],[2,6]],[9,5]]]").unwrap();
    let num2 = SnailfishNumber::from_str("[7,[[[3,7],[4,3]],[[6,3],[8,8]]]]").unwrap();

    let sum = num1 + num2;

    let result =
        SnailfishNumber::from_str("[[[[4,0],[5,4]],[[7,7],[6,0]]],[[8,[7,7]],[[7,9],[5,0]]]]")
            .unwrap();
    assert!(sum == result);
}

#[test]
fn test_magnitude() {
    let magnitude = |s: &str| SnailfishNumber::from_str(s).unwrap().magnitude();

    assert_eq!(
        magnitude("[[[[6,6],[7,6]],[[7,7],[7,0]]],[[[7,7],[7,7]],[[7,8],[9,9]]]]"),
        4140
    );
    assert_eq!(magnitude("[[1,2],[[3,4],5]]"), 143);
    assert_eq!(magnitude("[[[[0,7],4],[[7,8],[6,0]]],[8,1]]"), 1384);
    assert_eq!(magnitude("[[[[1,1],[2,2]],[3,3]],[4,4]]"), 445);
}

#[test]
fn test_addition_steps() {
    let test_cases = [
        (
            "[[[0,[4,5]],[0,0]],[[[4,5],[2,6]],[9,5]]]",
            "[7,[[[3,7],[4,3]],[[6,3],[8,8]]]]",
            "[[[[4,0],[5,4]],[[7,7],[6,0]]],[[8,[7,7]],[[7,9],[5,0]]]]",
        ),
        (
            "[[[[4,0],[5,4]],[[7,7],[6,0]]],[[8,[7,7]],[[7,9],[5,0]]]]",
            "[[2,[[0,8],[3,4]]],[[[6,7],1],[7,[1,6]]]]",
            "[[[[6,7],[6,7]],[[7,7],[0,7]]],[[[8,7],[7,7]],[[8,8],[8,0]]]]",
        ),
        (
            "[[[[6,7],[6,7]],[[7,7],[0,7]]],[[[8,7],[7,7]],[[8,8],[8,0]]]]",
            "[[[[2,4],7],[6,[0,5]]],[[[6,8],[2,8]],[[2,1],[4,5]]]]",
            "[[[[7,0],[7,7]],[[7,7],[7,8]]],[[[7,7],[8,8]],[[7,7],[8,7]]]]",
        ),
        (
            "[[[[7,0],[7,7]],[[7,7],[7,8]]],[[[7,7],[8,8]],[[7,7],[8,7]]]]",
            "[7,[5,[[3,8],[1,4]]]]",
            "[[[[7,7],[7,8]],[[9,5],[8,7]]],[[[6,8],[0,8]],[[9,9],[9,0]]]]",
        ),
        (
            "[[[[7,7],[7,8]],[[9,5],[8,7]]],[[[6,8],[0,8]],[[9,9],[9,0]]]]",
            "[[2,[2,2]],[8,[8,1]]]",
            "[[[[6,6],[6,6]],[[6,0],[6,7]]],[[[7,7],[8,9]],[8,[8,1]]]]",
        ),
        (
            "[[[[6,6],[6,6]],[[6,0],[6,7]]],[[[7,7],[8,9]],[8,[8,1]]]]",
            "[2,9]",
            "[[[[6,6],[7,7]],[[0,7],[7,7]]],[[[5,5],[5,6]],9]]",
        ),
        (
            "[[[[6,6],[7,7]],[[0,7],[7,7]]],[[[5,5],[5,6]],9]]",
            "[1,[[[9,3],9],[[9,0],[0,7]]]]",
            "[[[[7,8],[6,7]],[[6,8],[0,8]]],[[[7,7],[5,0]],[[5,5],[5,6]]]]",
        ),
        (
            "[[[[7,8],[6,7]],[[6,8],[0,8]]],[[[7,7],[5,0]],[[5,5],[5,6]]]]",
            "[[[5,[7,4]],7],1]",
            "[[[[7,7],[7,7]],[[8,7],[8,7]]],[[[7,0],[7,7]],9]]",
        ),
        (
            "[[[[7,7],[7,7]],[[8,7],[8,7]]],[[[7,0],[7,7]],9]]",
            "[[[[4,2],2],6],[8,7]]",
            "[[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]]",
        ),
    ];

    for (i, (num1, num2, expected)) in test_cases.into_iter().enumerate() {
        let num1 = SnailfishNumber::from_str(num1).unwrap();
        let num2 = SnailfishNumber::from_str(num2).unwrap();
        let expected = SnailfishNumber::from_str(expected).unwrap();

        let sum = num1 + num2;
        assert_eq!(sum, expected, "{}", i);
    }
}

fn add_all(nums: &str) -> SnailfishNumber {
    let numbers = nums
        .lines()
        .map(|num| SnailfishNumber::from_str(num).unwrap())
        .collect_vec();

    numbers.into_iter().fold1(std::ops::Add::add).unwrap()
}

#[test]
fn test_multi_addition() {
    let result = add_all(
        "[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]
[[[5,[2,8]],4],[5,[[9,9],0]]]
[6,[[[6,2],[5,6]],[[7,6],[4,7]]]]
[[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]
[[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]
[[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]
[[[[5,4],[7,7]],8],[[8,3],8]]
[[9,3],[[9,9],[6,[4,9]]]]
[[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]
[[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]",
    );

    let expected =
        SnailfishNumber::from_str("[[[[6,6],[7,6]],[[7,7],[7,0]]],[[[7,7],[7,7]],[[7,8],[9,9]]]]")
            .unwrap();

    assert_eq!(result, expected);
}